import numpy as np
import scipy.io as sio
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from models import transformer as transformer
from models import SSRN
from models import CNN
import utils
from utils import recorder
from evaluation import HSIEvaluation
from copy import deepcopy
from utils import device


class Tent(nn.Module):
    """Tent adapts a model by entropy minimization during testing."""

    def __init__(self, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "Tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        # Initialize the model and optimizer states
        self.model_state, self.optimizer_state = self.copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        """Forward pass with adaptation."""
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on a batch of data by minimizing entropy."""
        # Forward pass
        outputs = self.model(x)

        # Extract logits if outputs is a tuple
        if isinstance(outputs, tuple):
            logits = outputs[0]  # Assuming the first element is the logits
        else:
            logits = outputs  # Fallback if it's not a tuple

        # Compute the entropy loss using logits
        loss = softmax_entropy(logits).mean(0)

        # Backpropagate the entropy loss to update the model
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return logits, loss

    def copy_model_and_optimizer(self, model, optimizer):
        """Copy the model and optimizer states for resetting after adaptation."""
        model_state = deepcopy(model.state_dict())
        optimizer_state = deepcopy(optimizer.state_dict())
        return model_state, optimizer_state

    def load_model_and_optimizer(self, model, optimizer, model_state, optimizer_state):
        """Restore the model and optimizer states from saved copies."""
        model.load_state_dict(model_state, strict=True)
        optimizer.load_state_dict(optimizer_state)


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Compute the entropy of the softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def monitor_layernorm_parameters(model):
    """
    Monitor LayerNorm parameters (scale and shift) in a Transformer model.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            print(f"LayerNorm {name} - scale (weight): {module.weight}, shift (bias): {module.bias}")


def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.

         Walk the model's modules and collect all batch normalization parameters.
         Return the parameters and their names.

         Note: other choices of parameterization are possible!
         """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
        if isinstance(m, nn.LayerNorm):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    print("params:-------------------")
    print(names)
    print("params:-------------------")
    return params, names


def configure_model(model_0):
    """
    Configure model for TENT: only update LayerNorm layers' scale (weight) and shift (bias).
    """
    # deepcopy to ensure original model is not modified
    model = deepcopy(model_0)
    model.requires_grad_(False)
    # configure norm for tent updates: enable grad + force batch statisics
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            m.requires_grad_(True)
            # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        if isinstance(m, nn.LayerNorm):
            m.requires_grad_(True)
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None



    return model


class BaseTrainer(object):
    def __init__(self, params) -> None:
        self.tent_model = None
        self.tent_net = None
        self.params = params
        self.net_params = params['net']
        self.train_params = params['train']
        self.train_sign = self.params.get('train_sign', 'train') # train, test or tent
        self.path_model_save = self.params.get('path_model_save', '')
        self.model_loaded = False
        self.device = device 
        self.evalator = HSIEvaluation(param=params)
        self.class_num = self.params['data'].get('num_classes')

        self.net = None
        self.criterion = None
        self.optimizer = None
        self.clip = 15
        self.unlabel_loader=None

        # init model and check if use tent mode. if tent, configure tent model.
        self.real_init()

        # test configure
        if self.train_sign == 'test':
            self.load_model(self.path_model_save)
        # tent configure
        if self.train_sign == 'tent':
            self.load_model(self.path_model_save)
            self.confiture_tent()

    def real_init(self):
        pass

    def get_loss(self, outputs, target):
        return self.criterion(outputs, target)
       
    def train(self, train_loader, unlabel_loader=None, test_loader=None):
        # if tent, skip train...
        if self.train_sign in ['test', 'tent']:
            print("%s model skip train.." % (self.train_sign))
            return True
        epochs = self.params['train'].get('epochs', 100)
        total_loss = 0
        epoch_avg_loss = utils.AvgrageMeter()
        max_oa = 0
        for epoch in range(epochs):
            self.net.train()
            epoch_avg_loss.reset()
            for i, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.net(data)
                loss = self.get_loss(outputs, target)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.clip)
                self.optimizer.step()
                # batch stat
                total_loss += loss.item()
                epoch_avg_loss.update(loss.item(), data.shape[0])
            recorder.append_index_value("epoch_loss", epoch + 1, epoch_avg_loss.get_avg())
            print('[Epoch: %d]  [epoch_loss: %.5f]  [all_epoch_loss: %.5f] [current_batch_loss: %.5f] [batch_num: %s]' % (epoch + 1,
                                                                             epoch_avg_loss.get_avg(), 
                                                                             total_loss / (epoch + 1),
                                                                             loss.item(), epoch_avg_loss.get_num()))
            # 一定epoch下进行一次eval
            if test_loader and (epoch+1) % 10 == 0:
                y_pred_test, y_test = self.test(test_loader)
                temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
                max_oa = max(max_oa, temp_res['oa'])
                recorder.append_index_value("train_oa", epoch+1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch+1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch+1, temp_res['kappa'])
                recorder.append_index_value("max_oa", epoch+1, max_oa)
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (epoch+1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))
        print('Finished Training')

        torch.save(self.net.state_dict(), self.path_model_save)
        print("model saved.")
        return True

    def final_eval(self, test_loader):
        if self.train_sign == 'tent':
            y_pred_test, y_test = self.test(test_loader)  # test first
            y_pred_test, y_test = self.test_tent(test_loader)   
        else:
            y_pred_test, y_test = self.test(test_loader)  # test change
        temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
        return temp_res

    def get_logits(self, output):
        if type(output) == tuple:
            return output[0]
        return output

    def load_model(self, model_path):
        if self.path_model_save == "":
            raise ValueError("tent model need model path.")
        else:
            print("load model from model_path: %s" % self.path_model_save)
        self.net.load_state_dict(torch.load(model_path))
        self.model_loaded = True
        return self.net

    def confiture_tent(self): 
        print("start to confiture tent model...")
        assert self.model_loaded == True
        self.tent_net = configure_model(self.net)  # self.net
        params, param_names = collect_params(self.tent_net)
        optimizer = optim.SGD(params, lr=0.0001, momentum=0.9)
        self.tent_model = Tent(self.tent_net, optimizer, steps=1, episodic=False)
        return True

    def test_tent(self, test_loader):
        """
            Test the model on the test set, applying TENT adaptation if tent_model is defined.
        """
        epochs = self.params['train'].get('epochs', 100)
        tent_epochs = 2
        epoch_avg_loss = utils.AvgrageMeter()
        total_loss = 0

        print("start to test in tent mode...")
        for epoch in range(tent_epochs):
            # self.tent_model.train()
            # model.eval()
            # self.tent_model.eval()
            epoch_avg_loss.reset()
            count = 0
            y_pred_test = 0
            y_test = 0
            for inputs, labels in test_loader:
                inputs = inputs.to(self.device)

                # Apply TENT adaptation if tent_model is defined
                if self.tent_model is not None:
                    logits, loss = self.tent_model(inputs)  # This is likely a tuple
                else:
                    # outputs = model(inputs)
                    raise ValueError('tent_model should not be None.')
                if len(logits.shape) == 1:
                    continue
                if count % 10 == 0:
                    print("tent train epoch=%s, tent_loss=%s"  % (epoch, loss.detach().cpu().numpy()))
                # print(logits)
                # Get predictions (argmax over the softmax outputs)
                preds = np.argmax(logits.detach().cpu().numpy(), axis=1)

                # Concatenate predictions and ground truth labels
                if count == 0:
                    y_pred_test = preds
                    y_test = labels
                    count = count + 1
                else:
                    y_pred_test = np.concatenate((y_pred_test, preds))
                    y_test = np.concatenate((y_test, labels))

            if test_loader and (epoch + 1) % 1 == 0:
                # monitor_layernorm_parameters(self.tent_model)  # 监控 LayerNorm 参数的变化
                class_num = self.params['data'].get('num_classes')
                temp_res = self.evalator.eval(y_test, y_pred_test, self.class_num)
                recorder.append_index_value("train_oa", epoch + 1, temp_res['oa'])
                recorder.append_index_value("train_aa", epoch + 1, temp_res['aa'])
                recorder.append_index_value("train_kappa", epoch + 1, temp_res['kappa'])
                print('[--TEST--] [Epoch: %d] [oa: %.5f] [aa: %.5f] [kappa: %.5f] [num: %s]' % (
                epoch + 1, temp_res['oa'], temp_res['aa'], temp_res['kappa'], str(y_test.shape)))

        return y_pred_test, y_test


    def test(self, test_loader):
        """
        provide test_loader, return test result(only net output)
        """
        model = self.net
        count = 0
        self.net.eval()
        y_pred_test = 0
        y_test = 0
        for inputs, labels in test_loader:
            inputs = inputs.to(self.device)
            logits = self.get_logits(model(inputs))
            if len(logits.shape) == 1:
                continue
            outputs = np.argmax(logits.detach().cpu().numpy(), axis=1)
            
            if count == 0:
                y_pred_test = outputs
                y_test = labels
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
                y_test = np.concatenate((y_test, labels))
        return y_pred_test, y_test


class TransformerTrainer(BaseTrainer):
    def __init__(self, params):
        super(TransformerTrainer, self).__init__(params)
        self.lr = None
        self.weight_decay = None

    def real_init(self):
        # net
        self.net = transformer.TransFormerNet(self.params).to(self.device)
        # loss
        self.criterion = nn.CrossEntropyLoss()
        # optimizer
        self.lr = self.train_params.get('lr', 0.001)
        self.weight_decay = self.train_params.get('weight_decay', 5e-3)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def get_loss(self, outputs, target):
        """
            A_vecs: [batch, dim]
            B_vecs: [batch, dim]
            logits: [batch, class_num]
        """
        logits = outputs
        
        loss_main = nn.CrossEntropyLoss()(logits, target) 

        return loss_main   

