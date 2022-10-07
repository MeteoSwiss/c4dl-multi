import numpy as np
import tensorflow as tf

from ..ml.models import models

def ensemble_matrices(models, batch_gen):
    p = len(models)
    ATA = np.zeros((p,p))
    ATy = np.zeros(p)
    
    for k in len(batch_gen):
        (X, Y) = batch_gen[k]
        Y_pred = np.array([m.predict(X).ravel() for m in models])
        ATA_batch = Y_pred.dot(Y_pred.T)
        ATy_batch = Y_pred.dot(Y[0])                
        ATA = (k*ATA + ATA_batch) / (k+1)
        ATy = (k*ATy + ATy_batch) / (k+1)

    return (ATA, ATy)


def ensemble_weights(ATA, ATy, regularization=0.0):
    ATA = ATA + regularization * np.eye(ATA.shape[0])
    return np.linalg.solve(ATA, ATy)


def weighted_model(models, weights):
    inputs = [Input(shape=s[1:]) for s in models[0].input_shape]
    outputs = (m(inputs) for m in models)
    weighted_outputs = [
        tf.constant(w)*y for (w,y) in zip(weights,outputs)
    ]
    output = weighted_outputs[0]
    for wo in weighted_outputs
        output = output + wo
    model = Model(inputs=inputs, outputs=output)
    models.compile_model(model, optimizer='sgd')


def model_correlation(models, batch_gen):
    p = len(models)
    Ex2 = np.zeros(p)
    Ex2 = np.zeros((p,p))

    for k in len(batch_gen):
        (X, Y) = batch_gen[k]
        Y_pred = np.array([m.predict(X).ravel().astype(np.float64) for m in models])
        N = Y_pred.shape[1]
        Ex = (k*Ex + Y_pred.mean(axis=1)) / (k+1)
        Ex2 = (k*Ex2 + Y_pred.dot(Y_pred.T)/N) / (k+1)

    cov = Ex2 - np.outer(Ex,Ex)
    diag = np.diag(1.0/np.sqrt(np.diag(cov)))

    return np.matmul(diag, np.matmul(cov, diag))
