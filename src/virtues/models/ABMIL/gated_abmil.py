import torch.nn as nn
import torch
import numpy as np
from loguru import logger
from tqdm import tqdm
import wandb

class GatedABMIL(nn.Module):

    def __init__(self, emb_dim, hidden_dim, num_heads=1, feature_extractor=None, classifier=None, learnable_values=False) -> None:
        super().__init__()

        self.V = nn.Linear(emb_dim, hidden_dim)
        self.U = nn.Linear(emb_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, num_heads)

        if learnable_values:
            self.value_proj = nn.Linear(emb_dim, emb_dim)
        else:
            self.value_proj = nn.Identity()
        
        self.num_heads = num_heads

        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
        else:
            self.feature_extractor = nn.Identity()

        if classifier is not None:
            self.classifier = classifier
        else:
            self.classifier = nn.Identity()

    def forward(self, x, mask=None):
        """
        x: input of size B x S x D
        mask: mask of size B x S indicating padding (0)
        """
        x = self.feature_extractor(x)

        v_x = self.V(x)
        u_x = self.U(x)

        v_x = nn.functional.tanh(v_x)
        u_x = nn.functional.sigmoid(u_x)

        h = v_x * u_x

        attn_scores = self.W(h) #B x S x H
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(2), -1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=1) # B x S x H
        attn_weights = attn_weights.transpose(1, 2) # B x H x S

        output = torch.bmm(attn_weights, self.value_proj(x)) # B x H x D
        output_flat = output.reshape(-1, self.num_heads * x.size(2))

        return self.classifier(output_flat), output_flat

    def compute_attention(self, x, mask=None, batched=True):
        """
        x: input of size B x S x D
        """
        if not batched:
            x = x.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)
        
        if x.dim() > 3:
            reshaped = True
            old_shape = x.shape
            x = x.reshape(x.size(0), -1, x.size(-1))
        else:
            reshaped = False

        x = self.feature_extractor(x)
        v_x = self.V(x)
        u_x = self.U(x)

        v_x = nn.functional.tanh(v_x)
        u_x = nn.functional.sigmoid(u_x)

        h = v_x * u_x

        attn_scores = self.W(h) #B x S x H
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask.unsqueeze(-1), -1e9)
        attn_weights = nn.functional.softmax(attn_scores, dim=1) # B x S x H
        attn_weights = attn_weights.transpose(1, 2) # B x H x S
        
        if reshaped:
            attn_weights = attn_weights.reshape(old_shape[0], self.num_heads, *old_shape[1:-1])
        if not batched:
            attn_weights = attn_weights.squeeze(0)

        return attn_weights


class GatedABMILClassifierWithValidation(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_heads=1, num_classes=2, patience=5, verbose=True, wandb_log=False, save_path="src/abmil_factory/expt/", name=None, device='cuda'):
        super().__init__()
        
        self.num_classes = num_classes

        if self.num_classes == 2:
            classifier = nn.Sequential(
                nn.Linear(num_heads * input_dim, self.num_classes-1),
            )
            
        else:
            classifier = nn.Sequential(
                nn.Linear(num_heads * input_dim, self.num_classes),
            )

        self.patience = patience
        self.patience_counter = 0
        self.model = GatedABMIL(input_dim, hidden_dim, num_heads=num_heads, classifier=classifier)
        self.verbose = verbose

        self.best_model = None

        self.save_path = save_path
        self.name = name
        
        if self.name is None:
            self.name = "gated_abmil"

        self.wandb_log = wandb_log

        self.device = device    

    def forward(self, x, mask=None):
        return self.model(x, mask=mask)

    def compute_attention(self, x, mask=None, batched=True):
        return self.model.compute_attention(x, mask=mask, batched=batched)
    

    @torch.no_grad()
    def valid_eval(self, valid_dl, loss_fn):
        self.eval()
        valid_loss = 0
        predictions = []
        ground_truth = []
        for batch in valid_dl:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            logits, _ = self.forward(bags, mask=masks)
            # if self.num_classes == 2:
            #     labels = labels.unsqueeze(1).float()

            # loss = loss_fn(logits, labels)

            if self.num_classes == 2:
                labels = labels.unsqueeze(1).float()
            # print(logits.shape, labels.shape)
            loss = loss_fn(logits, labels)
            if self.num_classes > 2:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = torch.sigmoid(logits).round().squeeze(1)
            
            predictions.append(preds.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

            valid_loss += loss.item()

            # print(logits)

        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)

        if self.num_classes > 2:
            accuracy = (predictions == ground_truth).mean()
        else:
            accuracy = (predictions == ground_truth.squeeze(-1)).mean()

        # accuracy = (predictions == ground_truth).mean()

        return valid_loss / len(valid_dl), accuracy
    

    @torch.no_grad()
    def get_outputs(self, valid_dl, loss_fn=None, load_best=True):

        if load_best:
            self.load_best_model()

        self.eval()
        valid_loss = 0
        outputs = []
        ground_truth = []
        predictions = []
        # all_pre_final_outputs = []
        for batch in valid_dl:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            logits, output_flat = self.forward(bags, mask=masks)
            # if self.num_classes == 2:
            #     labels = labels.unsqueeze(1).float()

            # loss = loss_fn(logits, labels)

            if self.num_classes == 2:
                labels = labels.unsqueeze(1).float()
            # print(logits.shape, labels.shape)
            # loss = loss_fn(logits, labels)
            if self.num_classes > 2:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = torch.sigmoid(logits).round().squeeze(1)
            
            predictions.append(preds.cpu().numpy())
            # ground_truth.append(labels.cpu().numpy())
            # all_pre_final_outputs.append(output_flat.cpu().numpy())
            outputs.append(logits.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

            # valid_loss += loss.item()

        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)
        # all_pre_final_outputs = np.concatenate(all_pre_final_outputs)

        outputs = np.concatenate(outputs)

        if self.num_classes == 2:
            ground_truth = ground_truth.squeeze(-1)

        

        # accuracy = (predictions == ground_truth).mean()

        # logger.info(f"Accuracy: {accuracy}")

        # return valid_loss / len(valid_dl), outputs, ground_truth
        return {
            'logits': outputs,
            'ground_truth': ground_truth,
            'predictions': predictions,
            # 'pre_final_outputs': all_pre_final_outputs
        }


    def train_model(self, train_dl, valid_dl, num_epochs, optimizer, loss_fn, test_dl=None, monitor="valid_loss"):
        """
        Trains the model. train_dl should return batches of the form (bags, masks, labels) where bags BxSxD, masks BxS, labels B
        """
        self.to(self.device)
        self.average_train_loss = []
        self.average_train_accuracy = []
        self.average_valid_loss = []
        self.average_valid_accuracy = []

        if test_dl is not None:
            self.average_test_loss = []
            self.average_test_accuracy = []

        
        self.best_val_acc = 0
        self.best_val_loss = np.inf


        for ep in (range(num_epochs)):
            if self.wandb_log:
                wandb_log_dict = {}

            train_loss = 0
            self.train()
            for batch in train_dl:
                bags, masks, labels = batch
                bags = bags.to(self.device)
                masks = masks.to(self.device)
                labels = labels.to(self.device)
                logits, _ = self.forward(bags, mask=masks)
                if self.num_classes == 2:
                    labels = labels.unsqueeze(1).float()
                # print(logits.shape, labels.shape)
                loss = loss_fn(logits, labels)
                # if self.num_classes > 2:
                #     preds = torch.argmax(logits, dim=1)
                # else:
                #     preds = torch.sigmoid(logits).round().squeeze(1)
                
                # predictions.append(preds.cpu().detach().numpy())
                # ground_truth.append(labels.cpu().detach().numpy())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            # accuracy = (np.concatenate(predictions) == np.concatenate(ground_truth)).mean()
            # self.average_train_accuracy.append(accuracy)
            self.average_train_loss.append(train_loss / len(train_dl))

            valid_loss, valid_accuracy = self.valid_eval(valid_dl, loss_fn)
            self.average_valid_accuracy.append(valid_accuracy)
            self.average_valid_loss.append(valid_loss)

            if test_dl is not None:
                test_loss, test_accuracy = self.valid_eval(test_dl, loss_fn)
                self.average_test_accuracy.append(test_accuracy)
                self.average_test_loss.append(test_loss)
                if self.verbose:
                    logger.info(f"Epoch: {ep} Train Loss: {train_loss / len(train_dl):.3f}, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.3f}, Test Loss: {test_loss:.3f}, Test Accuracy: {test_accuracy:.3f}")
            else:
                if self.verbose:
                    logger.info(f"Epoch: {ep} Train Loss: {train_loss / len(train_dl):.3f}, Valid Loss: {valid_loss:.3f}, Valid Accuracy: {valid_accuracy:.3f}")
                

            if monitor == "valid_accuracy":

                if valid_accuracy > self.best_val_acc:
                    self.best_val_acc = valid_accuracy
                    self.best_val_loss = valid_loss
                    self.best_model = self.model.state_dict().copy()

                    torch.save(self.best_model, f"{self.save_path}/{self.name}.pt")

                    self.patience_counter = 0
                else:
                    self.patience_counter += 1

            elif monitor == "valid_loss":
                if valid_loss < self.best_val_loss:
                    self.best_val_acc = valid_accuracy
                    self.best_val_loss = valid_loss
                    self.best_model = self.model.state_dict().copy()
                    torch.save(self.best_model, f"{self.save_path}/{self.name}.pt")
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
            
            else:
                raise ValueError("monitor must be either valid_accuracy or valid_loss")
            
            if self.wandb_log:
                wandb_log_dict["epoch"] = ep
                wandb_log_dict["train_loss"] = train_loss / len(train_dl)
                wandb_log_dict["valid_loss"] = valid_loss
                wandb_log_dict["valid_accuracy"] = valid_accuracy
                wandb_log_dict["best_valid_loss"] = self.best_val_loss
                wandb_log_dict["best_valid_accuracy"] = self.best_val_acc
                if test_dl is not None:
                    wandb_log_dict["test_loss"] = test_loss
                    wandb_log_dict["test_accuracy"] = test_accuracy

                wandb.log(wandb_log_dict)


            if self.patience_counter >= self.patience:

                logger.info(f"Early stopping with best {monitor} @ loss: {self.best_val_loss:.3f}, acc: {self.best_val_acc:.3f}")
                break

            # early stopping

    def load_best_model(self):
        self.model.load_state_dict(torch.load(f"{self.save_path}/{self.name}.pt"))



    def finetune(self, train_dl, valid_dl, num_epochs, optimizer, loss_fn, monitor="valid_loss"):
        self.load_best_model()
        self.train_model(train_dl, valid_dl, num_epochs, optimizer, loss_fn, monitor=monitor)



    @torch.no_grad()
    def compute_predictions_and_truth(self, dataloader, loss_fn):
        """
        dataloader should return batches of the form (bags, masks, labels) where bags BxSxD, masks BxS, labels B
        Returns predictions and ground truth as numpy 1D arrays
        """
        self.to(self.device)
        predictions = []
        ground_truth = []

        test_loss = 0
        for batch in dataloader:
            bags, masks, labels = batch
            bags = bags.to(self.device)
            masks = masks.to(self.device)
            labels = labels.to(self.device)
            with torch.no_grad():
                logits = self.forward(bags, mask=masks)
            # if self.num_classes == 2:
                # labels = labels.unsqueeze(1).float()
            loss = loss_fn(logits, labels)
            if self.num_classes > 2:
                preds = torch.argmax(logits, dim=1)
            else:
                preds = torch.sigmoid(logits).round().squeeze(1)
            
            predictions.append(preds.cpu().numpy())
            ground_truth.append(labels.cpu().numpy())

            test_loss += loss.item()
            
        predictions = np.concatenate(predictions)
        ground_truth = np.concatenate(ground_truth)

        return predictions, ground_truth, test_loss
    


    def plot_loss_acc(self, save=False, name=""):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].plot(self.average_train_loss, label="Train Loss")
        ax[0].plot(self.average_valid_loss, label="Valid Loss")
        if hasattr(self, 'average_test_loss'):
            ax[0].plot(self.average_test_loss, label="Test Loss")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")
        ax[0].legend()

        ax[1].plot(self.average_train_accuracy, label="Train Accuracy")
        ax[1].plot(self.average_valid_accuracy, label="Valid Accuracy")
        if hasattr(self, 'average_test_accuracy'):
            ax[1].plot(self.average_test_accuracy, label="Test Accuracy")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy")
        ax[1].legend()


        if save:
            plt.savefig(f"/tmp/{name}_loss_acc_plot.png")
        else:
            plt.show()


