import random
sample_text = """We are about to study the idea of a computational process.
            Computational processes are abstract beings that inhabit computers .
            As they evolve , processes manipulate other abstract things called data .
            The evolution of a process is directed by a pattern of rules
            called a program . People create programs to direct processes . In effect ,
            we conjure the spirits of the computer with our spells .""".split()

# print(set(sample_text))

data = []
window_size = 2

for i in range(window_size, len(sample_text) - window_size):
    context = list(sample_text[i + j] for j in range(-window_size, window_size+1) if j != 0)
    target = sample_text[i]
    data.append((context, target))

print(data)

random.shuffle(sample_text)

# print(sample_text)
