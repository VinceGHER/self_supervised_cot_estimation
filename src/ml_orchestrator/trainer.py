import torch
import wandb
from torch.utils.data import DataLoader
import torch.optim as optim
from torcheval.metrics import R2Score, MeanSquaredError
import torch.multiprocessing as mp
import yaml
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from ..models.model_builder import model_builder

from .metrics.MAE_metric import MAEMetric

from .loss.loss_builder import loss_builder

from ..visualization.plotter import Plotter

from ..ml_orchestrator.dataset import COTDataset 

from ..tools import check_file_path
from .transforms.transforms_builder import TransformBuilder

class Trainer:
    def __init__(self,config,rank):
        self.device = config['ml_orchestrator']['device']
        self.r2_score = R2Score(device=self.device)
        self.mse = MeanSquaredError(device=self.device)
        self.mae = MAEMetric()

        self.rank = rank
        self.config = config



    def train_model(self,train_loader, valid_loader, model, criterion, optimizer, scheduler, num_epochs=25):

        try:
            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                if self.config['ml_orchestrator']['distributed_training']:
                    train_loader.sampler.set_epoch(epoch)
                for i,batch in enumerate(train_loader):
                    inputs = batch['image'].to(self.device)
                    depth = batch['depth'].to(self.device)
                    masks = batch['mask'].to(self.device)
                    confidence = batch['confidence'].to(self.device)

                    optimizer.zero_grad()

                    outputs = model(inputs,depth)

                    loss_total = criterion(outputs, masks, confidence,epoch)

                
                    loss_total.backward()
                    optimizer.step()
                    running_loss += loss_total.item()
                    if i == 0 and self.rank == 0 and epoch % self.config['ml_orchestrator']['valid_epoch'] == 0:
                        plotter = Plotter(config=self.config)

                        plot_image = plotter.create_image(
                            batch,
                            outputs.detach() if type(outputs) == torch.Tensor else outputs[0].detach(),
                        )


                        wandb.log({
                            f"plot_training": wandb.Image(plot_image),

                        }, step=epoch)

                loss = running_loss/len(train_loader)

                print(f"[GPU {self.rank}] Epoch {epoch+1}/{num_epochs}, Loss: {loss}")
                if self.rank == 0:
                    if epoch % self.config['ml_orchestrator']['valid_epoch'] == 0:
                        # Validation step
                        valid_loss, valid_metrics = self.validate_model(
                            valid_loader, 
                            model, 
                            criterion,
                            epoch,
                            name="valid_manually_labelled",
                            device=self.device)
                    

                        if scheduler is not None:
                            scheduler.step(valid_loss)  # Update the learning rate

                        metrics = {
                            "loss": loss,
                            "val_loss": valid_loss, 
                            "epoch": epoch,
                            **valid_metrics
                        }

                        if scheduler is not None:
                            metrics.update({'scheduler':scheduler.get_last_lr()[0]})
                        wandb.log(metrics, step=epoch)
                    else:
                        wandb.log({"loss": loss}, step=epoch)
                


        except KeyboardInterrupt:
            print('Training interrupted.')

        finally:
            print('Finished Training')
            if self.rank == 0:
                if self.config['ml_orchestrator']['distributed_training']:
                    torch.save(model.module.state_dict(), "trained_model.pth")
                else:
                    torch.save(model.state_dict(), "trained_model.pth")
                # save config as config using yaml
                with open("config.yaml", "w") as f:
                    yaml.dump(wandb.config.as_dict(), f)

            
    def validate_model(self,loader, model, criterion,epoch,name,device="cuda"):
        model.eval()
        running_val_loss = 0.0

        self.r2_score.reset()   
        self.mse.reset()    
        self.mae.reset()

        with torch.no_grad():
            for i,batch in enumerate(loader):
                inputs = batch['image'].to(device)
                depth = batch['depth'].to(device)
                masks = batch['mask'].to(device)
                confidence = batch['confidence'].to(device)

                outputs = model(inputs,depth)
                
                loss_total = criterion(outputs, masks, confidence,0 if not self.config['confidence'] else epoch)
                if not self.config['confidence']:
                    flat__outputs = outputs[0].reshape(-1) 
                    # outputs[1] = torch.where(masks >= self.config['cot']['wall_cot'], torch.zeros_like(outputs[1]), outputs[1])
                    flat_masks = outputs[1].reshape(-1)
                else:
                    flat__outputs = outputs.reshape(-1)
                    flat_masks = masks.reshape(-1)

                self.r2_score.update(flat__outputs, flat_masks)
                self.mse.update(flat__outputs, flat_masks)  
                self.mae.update(flat__outputs, flat_masks)

                running_val_loss += loss_total.item()

                if i == 0 and self.rank == 0:
                    plotter = Plotter(config=self.config)

                    plot_image = plotter.create_image(
                        batch,
                        outputs if type(outputs) == torch.Tensor else outputs[0],
                    )


                    wandb.log({
                        f"plot_{name}": wandb.Image(plot_image),

                    }, step=epoch)

        valid_r2 = self.r2_score.compute()
        valid_mse = self.mse.compute()
        valid_mae = self.mae.compute()

        validation_loss = running_val_loss/len(loader)
        print(f"Validation Loss {name}: {validation_loss}")

        return validation_loss, {
                f"valid_r2_{name}": valid_r2, 
                f"valid_mse_{name}": valid_mse, 
                f"valid_mae_{name}": valid_mae
            }
    

def print_stats(train_dataset, valid_manually_labelled_dataset):
        print("---------------------------------")
        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(valid_manually_labelled_dataset)}")
        print("---------------------------------")

def main(config=None):
    wandb.init(project=config['project_name'], name=config['exp_name'], config=config)
    
    if config['ml_orchestrator']['distributed_training']:
        world_size = torch.cuda.device_count()
        print(f"World size: {world_size}")
        mp.spawn(train, args=(config,world_size), nprocs=world_size)
    else:
        train(0,config,1)

def train(rank,config,world_size):
    
    if config['ml_orchestrator']['distributed_training']:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    if rank == 0:
        wandb.init(config=config,project=config['project_name'],name=config['exp_name'])
    transform_builder = TransformBuilder(config['transforms'])

    dataset_folder = check_file_path("datasets",config['ml_orchestrator']['dataset_name'])
    
    train_dataset = COTDataset(
        confidence=config['confidence'],
        root_dir=check_file_path(dataset_folder,"train"), 
        transform_input=transform_builder.build_transforms_inputs(),
        transform_common=transform_builder.build_transform_common(), 
        config=config, 
    )
    
    valid_manually_labelled_dataset = COTDataset(
        confidence=config['confidence'],
        root_dir=check_file_path(dataset_folder,"valid_manually_labelled"), 
        transform_input=transform_builder.build_transforms_inputs_validation(),
        transform_common=transform_builder.build_transform_common_validation(), 
        config=config, 
    )
    if config['ml_orchestrator']['distributed_training']:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['ml_orchestrator']['batch_size'], 
            shuffle=False,
            pin_memory=True,
            persistent_workers=config['ml_orchestrator']['persistent_workers'],
            num_workers=config['ml_orchestrator']['num_workers'],
            sampler=DistributedSampler(train_dataset)
        )
        valid_loader = DataLoader(
            valid_manually_labelled_dataset, 
            batch_size=config['ml_orchestrator']['batch_size'], 
            shuffle=True,
            pin_memory=True,
            num_workers=0,
        )
    else:
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config['ml_orchestrator']['batch_size'], 
            shuffle=True,
            pin_memory=True,
            persistent_workers=config['ml_orchestrator']['persistent_workers'],
            num_workers=config['ml_orchestrator']['num_workers']
        )
        valid_loader = DataLoader(
            valid_manually_labelled_dataset, 
            batch_size=config['ml_orchestrator']['batch_size'], 
            shuffle=True,
            pin_memory=True,
            persistent_workers=config['ml_orchestrator']['persistent_workers'],
            num_workers=config['ml_orchestrator']['num_workers']
        )

    print("Data loaders created")
    print_stats(train_dataset, valid_manually_labelled_dataset)
    
    plotter = Plotter(config=config)

    if rank == 0:
        for batch in train_loader:
            # limit batch to 4
            batch = {k: v[:4] for k, v in batch.items()}
            plotter.plot_batch_dataset(batch)
            break

    model = model_builder(config['model_builder'])
    model.to(config['ml_orchestrator']['device'])

    if config['ml_orchestrator']['distributed_training']:
        model = DDP(model, device_ids=[rank])
    
    criterion = loss_builder(config)

    if config['ml_orchestrator']['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config['ml_orchestrator']['learning_rate'])
    elif config['ml_orchestrator']['optimizer'] == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=config['ml_orchestrator']['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=config['ml_orchestrator']['learning_rate'], momentum=0.9)


    trainer = Trainer(config=config,rank=rank)
    trainer.train_model(train_loader, valid_loader, model, criterion, optimizer,None, num_epochs=config['ml_orchestrator']['epochs'])

    if rank == 0:
        artifact = wandb.Artifact(name="saved_model", type="model")
        artifact.add_file("trained_model.pth")
        artifact.add_file("config.yaml")
        wandb.log_artifact(artifact)
        wandb.finish()
    if config['ml_orchestrator']['distributed_training']:
        destroy_process_group()