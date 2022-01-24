
class SplitDataLoader:
    """
    This class distributes each sample among different workers.
    It returns a dictionary with key as data party's id and
    value as a pointer to the list of data batches at party's
    location.

    example:
    >>> from splitnn_dataloader import SplitDataLoader
    >>> splitnn_trainloader = SplitDataLoader(data_parties=data_parties, data_loader=trainloader)
    >>> splitnn_trainloader.data_pointer[1]['active_party'].shape, obj.data_pointer[1]['passive_party'].shape
     (torch.Size([64, 10]), torch.Size([64, 10]))
    """

    def __init__(self, data_parties, data_loader):
        """
         Args:
          data_parties: tuple of data parties
          data_loader: torch.utils.data.DataLoader
        """
        self.data_parties = data_parties
        self.data_loader = data_loader
        self.no_of_parties = len(data_parties)
        self.data_pointer = []
        self.labels = []

        """
            self.data_pointer:  list of dictionaries where 
            (key, value) = (id of the data holder, a pointer to the list of batches at that data holder).
            example:
            self.data_pointer  = [
                                    {"active_party": pointer_to_active_batch1, "passive_party": pointer_to_passive_batch1},
                                    {"active_party": pointer_to_active_batch2, "passive_party": pointer_to_passive_batch2},
                                    ...
                                 ]
        """

        # iterate over each batch of dataloader for split sample and send to VirtualWorker
        for samples, labels in self.data_loader:
            curr_data_dict = {}
            # calculate the feature number for each party according to the no. of workers
            feature_num_per_party = samples.shape[-1] // self.no_of_parties
            self.labels.append(labels)

            # iterate over each worker for distributing current batch of the self.data_loader
            for i, party in enumerate(self.data_parties[:-1]):
                # split the samples and send it to VirtualWorker (which is supposed to be an party or client)
                sample_part_ptr = samples[:, feature_num_per_party * i:feature_num_per_party * (i + 1)].send(
                    party
                )
                curr_data_dict[party.id] = sample_part_ptr

            # repeat same for the remaining part of the samples
            last_party = self.data_parties[-1]
            last_part_ptr = samples[:, feature_num_per_party * (i + 1):].send(last_party)
            curr_data_dict[last_party.id] = last_part_ptr
            self.data_pointer.append(curr_data_dict)
            
    def __iter__(self):
        for data_ptr, label in zip(self.data_pointer[:-1], self.labels[:-1]):
            yield data_ptr, label
            
    def __len__(self):
        return len(self.data_loader) - 1
