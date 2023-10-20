'''
This script contains Base R implementations of varimax and promax rotations
as well as the psych package implementation of varimax and promax rotations.
the psych package uses the GPArotation package implementation of varimax and promax
evaluation of their performamce in terms of speed indicates that the base R implementation 
is faster than the psych package implementation.
'''
### base R implementation of promax
promax <-function (x, m = 4) 
{
  if (ncol(x) < 2) 
    return(x)
  dn <- dimnames(x)
  xx <- varimax(x)
  x <- xx$loadings
  Q <- x * abs(x)^(m - 1)
  U <- lm.fit(x, Q)$coefficients
  d <- diag(solve(t(U) %*% U))
  U <- U %*% diag(sqrt(d))
  dimnames(U) <- NULL
  z <- x %*% U
  U <- xx$rotmat %*% U
  dimnames(z) <- dn
  class(z) <- "loadings"
  list(loadings = z, rotmat = U)
}

### base R implementation of varimax
varimax <- function(x, normalize = TRUE, eps = 1e-05) 
{
  nc <- ncol(x)
  if (nc < 2) 
    return(x)
  if (normalize) {
    sc <- sqrt(drop(apply(x, 1L, function(x) sum(x^2))))
    x <- x/sc
  }
  p <- nrow(x)
  TT <- diag(nc)
  d <- 0
  for (i in 1L:1000L) {
    z <- x %*% TT
    B <- t(x) %*% (z^3 - z %*% diag(drop(rep(1, p) %*% z^2))/p)
    sB <- La.svd(B)
    TT <- sB$u %*% sB$vt
    dpast <- d
    d <- sum(sB$d)
    if (d < dpast * (1 + eps)) 
      break
  }
  z <- x %*% TT
  if (normalize) 
    z <- z * sc
  dimnames(z) <- dimnames(x)
  class(z) <- "loadings"
  list(loadings = z, rotmat = TT)
}


View(varimax)
View(fa)
View(fac)
View(GPArotation::Varimax)
View(GPArotation::GPFRSorth)

#############################################
#### implementing rotations using psych package
#############################################
library(psych)
library(mvtnorm)

# Set a random seed for reproducibility
set.seed(123)

# Simulate data with 3 factors and 10 variables
n <- 200  # Number of observations
p <- 10   # Number of variables
factors <- 3  # Number of factors

# Create a random correlation matrix with 3 factors
# Generate random factor loadings
loading_matrix <- matrix(rnorm(p * factors), nrow = p, ncol = factors)

# Generate random error variances
error_variances <- runif(p, 0.5, 2.0)

# Create a random correlation matrix
cor_matrix <- loading_matrix %*% t(loading_matrix) + diag(error_variances)

# Generate data based on the correlation matrix
simulated_data <- rmvnorm(n, mean = rep(0, p), sigma = cor_matrix)

# Factor analysis with rotations
fa_varimax <- fa(simulated_data, nfactors = factors, rotate = "varimax")
fa_quartimax <- fa(simulated_data, nfactors = factors, rotate = "quartimax")
fa_equamax <- fa(simulated_data, nfactors = factors, rotate = "equamax")
fa_promax <- fa(simulated_data, nfactors = factors, rotate = "promax")
fa_oblimin <- fa(simulated_data, nfactors = factors, rotate = "oblimin")

# Print the factor loadings for each rotation
print("Varimax Rotation:")
print(fa_varimax$loadings)

print("Quartimax Rotation:")
print(fa_quartimax$loadings)

print("Equamax Rotation:")
print(fa_equamax$loadings)

print("Promax Rotation:")
print(fa_promax$loadings)

print("Direct Oblimin Rotation:")
print(fa_oblimin$loadings)

#############################################
#### checking rotation performance using psych package
#############################################

# Simulate data with 3 factors and 10 variables
n <- 200  # Number of observations
p <- 10   # Number of variables
factors <- 3  # Number of factors

# Create a random correlation matrix with 3 factors
cor_matrix <- simulateData(factors = factors, variables = p)
# Generate data based on the correlation matrix
simulated_data <- simulate(cor_matrix = cor_matrix, sampleSize = n)

# Initialize an empty data frame to store evaluation metrics
results_df <- data.frame(
  Method = character(),
  VarianceExplained = numeric(),
  ChiSquare = numeric(),
  RMSEA = numeric()
)

# Define a list of rotation methods to evaluate
rotation_methods <- c("varimax", "quartimax", "equamax", "promax", "oblimin")

# Loop through each rotation method and assess the results
for (method in rotation_methods) {
  # Perform factor analysis with the current rotation method
  fa_result <- fa(simulated_data, nfactors = factors, rotate = method)

  # Calculate the proportion of variance explained by the factors
  variance_explained <- sum(fa_result$Vaccounted)

  # Calculate the chi-square goodness-of-fit test statistic
  chi_square <- fa_result$chisq

  # Calculate the root mean square error of approximation (RMSEA)
  rmsea <- fa_result$RMSEA['RMSEA']

  # Store the results in the data frame
  results_df <- rbind(results_df, data.frame(Method = method, VarianceExplained = variance_explained, ChiSquare = chi_square, RMSEA = rmsea))
}

# Print the evaluation results
print(results_df)




