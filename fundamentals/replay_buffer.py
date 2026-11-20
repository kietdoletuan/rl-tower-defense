import random

class buffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.storage = [None] * capacity
        self.stored = 0
        self.head = 0

    def push(self, state, action, reward, next_state, terminated):
        if self.head == self.capacity:
            self.head = 0
        self.storage[self.head] = ([state, action, reward, next_state, terminated])
        self.head += 1
        if self.stored != self.capacity:
            self.stored += 1

    def sample(self, batch_size):
        indices = random.sample(range(self.stored), k=batch_size)
        state, action, reward, next_state, terminated = [], [], [], [], []

        for transition in [self.storage[i] for i in indices]:
            state.append(transition[0])
            action.append(transition[1])
            reward.append(float(transition[2]))
            next_state.append(transition[3])
            terminated.append(transition[4])

        return (state, action, reward, next_state, terminated)
        
    def __len__(self):
        return self.stored 

if __name__ == "__main__":
    random.seed(100)
    the_buffer = buffer(3)
    transitions = [[0,0,1,1,0],[1,1,2,2,0],[10,0,0.5,11,0],[2,0,3,3,0],[3,1,4,0,1]]
    for transition in transitions:
        the_buffer.push(*transition)
    print(the_buffer.sample(3))
    print(len(the_buffer))