import numpy as np
import torch
        
class ConvNet(torch.nn.Module):
    
    def __init__(self, input_shape, n_hiddens_per_conv_layer, n_hiddens_per_fc_layer, n_outputs, 
                 patch_size_per_conv_layer, stride_per_conv_layer, activation_function='tanh', device='cpu'):
        
        super().__init__()
        
        self.device = device

        n_conv_layers = len(n_hiddens_per_conv_layer)
        if (len(patch_size_per_conv_layer) != n_conv_layers or
            len(stride_per_conv_layer) != n_conv_layers):
            raise Exception('The lengths of n_hiddens_per_conv_layer, patch_size_per_conv_layer, and stride_per_conv_layer must be equal.')
        
        self.activation_function = torch.tanh if activation_function == 'tanh' else torch.relu
        
        # Create all convolutional layers
        # First argument to first Conv2d is number of channels for each pixel.
        # Just 1 for our grayscale images.
        n_in = input_shape[0]
        input_hw = input_shape[1]  # = input_shape[2]
        self.conv_layers = torch.nn.ModuleList()
        for nh, patch_size, stride in zip(n_hiddens_per_conv_layer,
                                          patch_size_per_conv_layer,
                                          stride_per_conv_layer):
            self.conv_layers.append( torch.nn.Conv2d(n_in, nh, kernel_size=patch_size, stride=stride) )
            conv_layer_output_hw = (input_hw - patch_size) // stride + 1
            input_hw = conv_layer_output_hw  # for next trip through this loop
            n_in = nh

        # Create all fully connected layers.  First must determine number of inputs to first
        # fully-connected layer that results from flattening the images coming out of the last
        # convolutional layer.
        n_in = input_hw ** 2 * n_in  # n_hiddens_per_fc_layer[0]
        self.fc_layers = torch.nn.ModuleList()
        for nh in n_hiddens_per_fc_layer:
            self.fc_layers.append( torch.nn.Linear(n_in, nh) )
            n_in = nh

        output_layer = torch.nn.Linear(n_in, n_outputs)
        self.fc_layers.append(output_layer)
        
        output_layer.weight.data[:] = 0.0
        output_layer.bias.data[:] = 0.0

        self.loss_trace = []
        self.accuracy_trace = []
        
        self.to(self.device)


    def _forward_all_outputs(self, X):
        n_samples = X.shape[0]
        Ys = [X]
        for conv_layer in self.conv_layers:
            Ys.append( self.activation_function(conv_layer(Ys[-1])) )

        for layeri, fc_layer in enumerate(self.fc_layers[:-1]):
            if layeri == 0:
                flattend_inputs = Ys[-1].reshape(n_samples, -1)
                Ys.append( self.activation_function(fc_layer(flattend_inputs)) )
            else:
                Ys.append( self.activation_function(fc_layer(Ys[-1])) )

        if len(self.fc_layers) == 1:  # no fully connected hidden layers
            flattend_inputs = Ys[-1].reshape(n_samples, -1)
            Ys.append(self.fc_layers[-1](flattend_inputs))
        else:
            Ys.append(self.fc_layers[-1](Ys[-1]))
        return Ys


    def _forward(self, X):
        Ys = self._forward_all_outputs(X)
        return Ys[-1]
    

    def to_torch(self, M, torch_type=torch.FloatTensor):
        if not isinstance(M, torch.Tensor):
            return torch.from_numpy(M).type(torch_type).to(self.device)
        return M
        
    def percent_correct(self, Y_classes, T):
        if isinstance(T, torch.Tensor):
            T = T.cpu().numpy()
        return (Y_classes == T).mean() * 100
    
    def train(self, Xtrain, Ttrain, batch_size, n_epochs, learning_rate, method='sgd', verbose=True,
              Xval=None, Tval=None):
        
        # Assuming Ttrain includes all possible class labels
        self.classes = np.unique(Ttrain)

        # Set data matrices to torch.tensors if not already.

        Xtrain = self.to_torch(Xtrain)
        Ttrain = self.to_torch(Ttrain, torch.LongTensor)
        Xval = self.to_torch(Xval) if Xval is not None else None
        Tval = self.to_torch(Tval, torch.LongTensor) if Tval is not None else None
        
        Xtrain.requires_grad_(True)

        if method == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        loss_f = torch.nn.CrossEntropyLoss(reduction='mean')
        
        for epoch in range(n_epochs):

            if batch_size == -1:
                num_batches = 1
            else:
                num_batches = Xtrain.shape[0] // batch_size

            loss_sum = 0
            class_train_sum = 0
            
            for k in range(num_batches):
                
                start = k * batch_size
                end = (k + 1) * batch_size
                X_batch = Xtrain[start:end, ...]
                T_batch = Ttrain[start:end, ...]
                
                Y = self._forward(X_batch)
                
                loss = loss_f(Y, T_batch)
                loss.backward()

                # Update parameters
                optimizer.step()
                optimizer.zero_grad()

                loss_sum += loss

                with torch.no_grad():
                    class_train_sum += self.percent_correct(self.use(X_batch)[0], T_batch)
                    
            self.loss_trace.append((loss_sum.item() / num_batches))
            percent_correct_train = class_train_sum / num_batches
            if Xval is not None:
                with torch.no_grad():
                    percent_correct_val = self.percent_correct(self.use(Xval)[0], Tval)
                self.accuracy_trace.append([percent_correct_train, percent_correct_val])
            else:
                self.accuracy_trace.append(percent_correct_train)
                
            if verbose and (epoch + 1) % (n_epochs // 10) == 0:
                print(method, 'Epoch', epoch + 1, 'Loss', self.loss_trace[-1])

        return self


    def _softmax(self, Y):
        '''Apply to final layer weighted sum outputs'''
        # Trick to avoid overflow
        maxY = torch.max(Y, axis=1)[0].reshape((-1,1))
        expY = torch.exp(Y - maxY)
        denom = torch.sum(expY, axis=1).reshape((-1, 1))
        Y = expY / denom
        return Y


    def use(self, X):
        # Set input matrix to torch.tensors if not already.
        with torch.no_grad():
            X = self.to_torch(X)
            Y = self._forward(X)
            probs = self._softmax(Y).cpu().numpy()
            classes = self.classes[np.argmax(probs, axis=1)]
            return classes, probs

    def get_loss_trace(self):
        return self.loss_trace

    def get_accuracy_trace(self):
        return self.accuracy_trace
