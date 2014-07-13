#Model Parameters
model.mu = 0
model.sigma = 1
model.nu = 0
model.tau = 1

#Particle Filter Parameters
filter.mu = model.mu
filter.sigma = model.sigma
filter.nu = model.nu
filter.tau = model.tau

#Model distributions
dInit <- function(x)
{
  return (dnorm(x, model.mu, model.sigma))
}


dPrior <- function(xold,xnew)
{
  return (dnorm(xold - xnew, model.mu, model.sigma))
}

dLikelihood <- function(x,y)
{
  return (dnorm(y - x, model.nu, model.tau))
}

#Proposal distributions
dInitProp <- function(x, y=NA)
{
  return (dnorm(x, filter.mu, filter.sigma))
}

rInitProp <- function(y=NA)
{
  return (rnorm(1, filter.mu, filter.sigma))
}

dProposal <- function(xold,xnew,y=NA)
{
  return (dnorm(xold - xnew, filter.mu, filter.sigma))
}

rProposal <- function(xold, y=NA)
{
  return (rnorm(1, xold + filter.mu, filter.sigma))
}