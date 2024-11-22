from scipy.stats import binom

# Parameters
n = 963  # Total number of trials
p = 0.71  # Probability of picking an apple
k = 456  # Number of successes (apples)

# Calculate the exact binomial probability
probability = binom.pmf(k, n, p)
print(probability)
