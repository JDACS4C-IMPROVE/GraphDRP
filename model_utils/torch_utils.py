import os
import torch
from torch_geometric.data import InMemoryDataset, DataLoader
from torch_geometric import data as DATA

# from models.gat import GATNet
# from models.gat_gcn import GAT_GCN
# from models.gcn import GCNNet
# from models.ginconv import GINConvNet

from .models.gat import GATNet
from .models.gat_gcn import GAT_GCN
from .models.gcn import GCNNet
from .models.ginconv import GINConvNet


class TestbedDataset(InMemoryDataset):
    def __init__(self,
                 root='/tmp',
                 dataset='davis',
                 xd=None,
                 xt=None,
                 y=None,
                 transform=None,
                 pre_transform=None,
                 smile_graph=None,
                 saliency_map=False):

        #root is required for save preprocessed data, default is '/tmp'
        super(TestbedDataset, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        self.saliency_map = saliency_map
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(
                self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(
                self.processed_paths[0]))
            self.process(xd, xt, y, smile_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])
        # dd = self.data
        # ss = self.slices
        # print(dd)
        # print(ss.keys())
        # print(dd.x.shape)
        # print(dd.c_size)
        print("PyTorch Dataset initialized.\n")

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    # Customize the process method to fit the task of drug-target affinity prediction
    # Inputs:
    # XD - list of SMILES, XT: list of encoded target (categorical or one-hot),
    # Y: list of labels (i.e. affinity)
    # Return: PyTorch-Geometric format processed data
    def process(self, xd, xt, y, smile_graph):
        assert (len(xd) == len(xt)
                and len(xt) == len(y)), "The three lists must be the same length!"
        data_list = []
        data_len = len(xd)
        for i in range(data_len):
            #print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]  # SMILES of a drug
            target = xt[i]  # omic vector of cell
            labels = y[i]   # response
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index = smile_graph[smiles]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            GCNData = DATA.Data(
                x=torch.Tensor(features),
                edge_index=torch.LongTensor(edge_index).transpose(1, 0),
                y=torch.FloatTensor([labels]))

            # require_grad of cell-line for saliency map
            if self.saliency_map == True:
                GCNData.target = torch.tensor([target],
                                              dtype=torch.float,
                                              requires_grad=True)
            else:
                GCNData.target = torch.FloatTensor([target])

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(GCNData)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])

    def getXD(self):
        return self.xd


def build_GraphDRP_dataloader(
        data_dir: str,
        data_fname: str,
        batch_size: int,
        shuffle: bool):
    """ Build a PyTorch data loader for GraphDRP.

    :params str datadir: Directory where `processed` folder containing
            processed data can be found.
    :params str datafname: Name of PyTorch processed data to read.
    :params int batch: Batch size for data loader.
    :params bool shuffle: Flag to specify if data is to be shuffled when
            applying data loader.

    :return: PyTorch data loader constructed.
    :rtype: DataLoader
    """
    if data_fname.endswith(".pt"):
        data_fname = data_fname[:-3] # TestbedDataset() appends this string with ".pt"
    dataset = TestbedDataset(root=data_dir, dataset=data_fname) # TestbedDataset() requires strings
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # PyTorch dataloader
    return data_loader


def train_epoch(model, device, train_loader, optimizer, loss_fn, epoch: int,
                log_interval: int, verbose=True):
    """Execute a training epoch (i.e. one pass through training set).

    :params DataLoader train_loader: PyTorch data loader with training data.
    :params int epoch: Current training epoch (for display purposes only).

    :return: Average loss for executed training epoch.
    :rtype: float
    """
    print("Training on {} samples...".format(len(train_loader.dataset)))
    model.train()
    # Below is the train() from the original GraphDRP model
    avg_loss = []
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        # breakpoint();
        # ll = torch.nn.functional.mse_loss(input=output,
        #                              target=data.y.view(-1, 1).float(),
        #                              size_average=None, reduce=None,
        #                              reduction='mean')
        # print(np.sum(np.isnan(output.cpu().detach().numpy())))
        # print(ll)
        loss = loss_fn(output, data.y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print(
            "Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                batch_idx * len(data.x),
                len(train_loader.dataset),
                100.0 * batch_idx / len(train_loader),
                loss.item(),
            )
        )
    return sum(avg_loss) / len(avg_loss)


def predicting(model, device, data_loader):
    """ Method to make predictions/inference.
    This is used in *train.py and *infer.py

    Parameters
    ----------
    model : pytorch model
        Model to evaluate.
    device : string
        Identifier for hardware that will be used to evaluate model.
    data_loader : pytorch data loader.
        Object to load data to evaluate.

    Returns
    -------
    total_labels: numpy array
        Array with ground truth.
    total_preds: numpy array
        Array with inferred outputs.
    """
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    print("Make prediction for {} samples...".format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output, _ = model(data)
            # Is this computationally efficient?
            total_preds = torch.cat((total_preds, output.cpu()), 0)  # preds to tensor
            total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)  # labels to tensor
    return total_labels.numpy().flatten(), total_preds.numpy().flatten()


def str2Class(str):
    """ Get model class from model name (str) """
    return globals()[str]()


def set_GraphDRP(params, device):
    """ Chooses the specific GraphDRP architecture and moves it to device """
    model = str2Class(params["model_arch"]).to(device)
    return model


def load_GraphDRP(params, modelpath, device):
    """ Load GraphDRP """
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! modelpath not found {modelpath}\n")
    model = str2Class(params["model_arch"]).to(device)
    model.load_state_dict(torch.load(modelpath))
    model.eval()
    return model


def determine_device(cuda_name_from_params):
    """Determine device to run PyTorch functions.

    PyTorch functions can run on CPU or on GPU. In the latter case, it
    also takes into account the GPU devices requested for the run.

    :params str cuda_name_from_params: GPUs specified for the run.

    :return: Device available for running PyTorch functionality.
    :rtype: str
    """
    cuda_avail = torch.cuda.is_available()
    print("GPU available: ", cuda_avail)
    if cuda_avail:  # GPU available
        # -----------------------------
        # CUDA device from env var
        cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
        if cuda_env_visible is not None:
            # Note! When one or multiple device numbers are passed via
            # CUDA_VISIBLE_DEVICES, the values in python script are reindexed
            # and start from 0.
            print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
            cuda_name = "cuda:0"
        else:
            cuda_name = cuda_name_from_params
        device = cuda_name
    else:
        device = "cpu"

    return device
