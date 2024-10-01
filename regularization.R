# Lab III - Regularization and dimension reduction ####

# Ridge regression ####
library(ElemStatLearn)

data("prostate")
names(prostate)

x<-prostate[,-ncol(prostate)]
p<-ncol(x)-1

# install.packages('glmnet')
library(glmnet)
x<-model.matrix(lpsa~.,x)[,-1]
y<-prostate$lpsa

grid<-10^seq(10,-2,length=100)
summary(grid)
ridge.mod<-glmnet(x,y,alpha=0,lambda = grid)

dim(coef(ridge.mod))
coef(ridge.mod)[,1]
coef(ridge.mod)[,100]

plot(apply(coef(ridge.mod)[-1,],2,function(x) sum(x^2)))

## 10-fold CV for ridge regression ####
n<-nrow(x)
set.seed(1234)
index<-sample(1:n,ceiling(n/2))
set.seed(1234)
cv.out<-cv.glmnet(x[index,],y[index],alpha=0) # alpha=0 does ridge regression
plot(cv.out)

cv.out$lambda
names(cv.out)
cv.out$lambda.min
cv.out$lambda.1se


best.lambda<-cv.out$lambda.min
pred.ridge<-predict(cv.out,s=best.lambda,newx=x[-index,],x=x[index,],y=y[index])
mean((y[-index]-pred.ridge)^2)

predict(cv.out,s=best.lambda,newx=x[-index,],x=x[index,],y=y[index],type="coefficient")

## Compare accuracy with Kinda OLS ####
pred.ols<-predict(cv.out,s=0,newx=x[-index,],x=x[index,],y=y[index],exact=T)
mean((y[-index]-pred.ols)^2)


# Lasso regression ####
# alpha=1
out.lasso<-glmnet(x[index,],y[index],alpha=1,lambda=grid)
plot(out.lasso)


## Choice of optimal lambda in 10-fold CV ####
set.seed(1234)
cv.out<-cv.glmnet(x[index,],y[index],alpha=1)
cv.out$lambda.min
cv.out$lambda.1se
plot(cv.out)

best.lambda<-cv.out$lambda.min
lasso.predict<-predict(cv.out,s=best.lambda,
                       newx=x[-index,],x=x[index,],y=y[index])
mean((y[-index]-lasso.predict)^2)

(beta.hat.lasso<-predict(cv.out,s=best.lambda,
        newx=x[-index,],x=x[index,],y=y[index],
        type="coefficient"))
# The optimal lambda (chosen in CV) returns a model with 
# 5 predictors out of 8  


# Dimension Reduction ####

## PCR - Principal Component Regression ####
library(ElemStatLearn)
data(prostate)
install.packages('pls')
library(pls)

set.seed(1234)
out.pcr<-pcr(lpsa~.,data=prostate[,-ncol(prostate)],
             scale=T,validation="CV")
summary(out.pcr)
validationplot(out.pcr)

# The PLS package allows for "selectNcomp"
# procedure to select the optimal no. of components

n.comp.os<-selectNcomp(out.pcr,method="onesigma",plot=T)
n.comp.r<-selectNcomp(out.pcr,method="randomization",plot=T)

## Validation set approach ####
set.seed(1234)
n<-nrow(prostate)
train<-sample(1:n,ceiling(n/2))
test<- -train

set.seed(1234)
pcr.out<-pcr(lpsa~.,data=prostate[,-ncol(prostate)],
             subset=train,scale=T,validation="CV")
validationplot(pcr.out)
selectNcomp(pcr.out,method="onesigma",plot=T)

pred.pcr<-predict(pcr.out,prostate[-train,1:8],
                  ncomp=3)
mean((prostate$lpsa[test]-pred.pcr)^2)


## PCR via eigen ####
x.pcs<-eigen(cor(prostate[train,1:8]))$vectors
train.x<-prostate[train,1:8]
test.x<-prostate[-train,1:8]
test.std<-test.x
for (j in 1:ncol(test.x)){
  test.std[,j]<-(test.x[,j]-mean(train.x[,j]))/sd(train.x[,j])
}

x.train<-scale(train.x,T,T)%*%x.pcs[,1:3] # projected onto the first 3 PCs
x.test<-as.matrix(test.std)%*%x.pcs[,1:3] # projected onto the first 3 PCs

df.pcs<-data.frame(y=c(prostate$lpsa[train],prostate$lpsa[-train]),
                   rbind(x.train,x.test))
out.lm<-lm(y~.,data=df.pcs[1:length(train),])
y.hat<-predict(out.lm,newdata = df.pcs[-c(1:length(train)),])
mean((df.pcs$y[-c(1:length(train))]-y.hat)^2)

## PCR via SVD ####
x.std<-scale(prostate[train,1:8],T,T)
svd.x<-svd(x.std)
x.pcs<-svd.x$v[,1:3]
xx.train<-x.std%*%x.pcs
xx.test<-as.matrix(test.std)%*%x.pcs

df.svd<-data.frame(y=c(prostate$lpsa[train],prostate$lpsa[-train]),
                   rbind(xx.train,xx.test))
out.svd<-lm(y~.,data=df.svd[1:length(train),])
yy.hat<-predict(out.svd,newdata=df.svd[-c(1:length(train)),])
mean((df.svd$y[-c(1:length(train))]-yy.hat)^2)
