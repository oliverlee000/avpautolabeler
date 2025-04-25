# Load necessary library
set.seed(42)

# Read the dataset
data <- read.csv("data_unsplit.csv")

# Shuffle the data
shuffled_data <- data[sample(nrow(data)), ]

# Calculate split sizes
n <- nrow(shuffled_data)
train_size <- floor(0.8 * n)
dev_size <- floor(0.1 * n)
test_size <- n - train_size - dev_size  # in case rounding makes it slightly off

# Split the data
train_data <- shuffled_data[1:train_size, ]
dev_data <- shuffled_data[(train_size + 1):(train_size + dev_size), ]
test_data <- shuffled_data[(train_size + dev_size + 1):n, ]

# Save to CSV
write.csv(train_data, "train.csv", row.names = FALSE)
write.csv(dev_data, "dev.csv", row.names = FALSE)
write.csv(test_data, "test.csv", row.names = FALSE)

cat("Split complete. Files saved as train.csv, dev.csv, and test.csv.\n")
