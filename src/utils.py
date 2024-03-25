class Memory(object):
    def __init__(self):
        self.examples = []
        self.labels = []
        self.tasks = []

    def append(self, example, label, task):
        self.examples.append(example)
        self.labels.append(label)
        self.tasks.append(task)

    def get_random_batch(self, batch_size, task_id=None):
        if task_id is None:
            permutations = np.random.permutation(len(self.labels))
            index = permutations[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
        else:
            index = [i for i in range(len(self.labels)) if self.tasks[i] == task_id]
            np.random.shuffle(index)
            index = index[:batch_size]
            mini_examples = [self.examples[i] for i in index]
            mini_labels = [self.labels[i] for i in index]
            mini_tasks = [self.tasks[i] for i in index]
        return torch.tensor(mini_examples), torch.tensor(mini_labels), torch.tensor(mini_tasks)

    def __len__(self):
        return len(self.labels)