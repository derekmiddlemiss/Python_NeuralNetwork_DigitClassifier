#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import random

class Neural_Network():
    def __init__(self, inputLayerSize, outputLayerSize, \
                 hiddenLayerSize):
        #Network hyperparameters - neurons per layer - **not altered by training**
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.num_params = inputLayerSize * hiddenLayerSize + \
                            hiddenLayerSize * outputLayerSize + hiddenLayerSize \
                            + outputLayerSize
        #--Weights--
        #w_ih - weights of synapses linking input -> hidden
        self.w_ih = np.random.randn( self.inputLayerSize, \
                                  self.hiddenLayerSize)
        #w_ho - weights of synapses linking hidden -> output
        self.w_ho = np.random.randn( self.hiddenLayerSize, \
                                  self.outputLayerSize)
        
        #--Biases--
        #b_h - biases of hidden layer
        self.b_h = np.random.randn( self.hiddenLayerSize )
        #b_o - biases of output layer
        self.b_o = np.random.randn( self.outputLayerSize )
        
    def forward_propagate(self, x):
        self.z_h = np.dot( x, self.w_ih ) + self.b_h
        self.a_h = self.sigmoid( self.z_h )
        self.z_o = np.dot( self.a_h, self.w_ho ) + self.b_o
        yEst = self.sigmoid( self.z_o )
        return yEst
                
    def sigmoid(self, z):
        return 1/( 1 + np.exp(-z) )
    
    def deriv_sigmoid(self,z):
        return np.exp(-z) / ( (1 + np.exp(-z)) ** 2 )
    
    def costFunction(self, x, y ):
        self.yEst = self.forward_propagate(x)
        sqErrors = ( self.yEst - y ) ** 2
        J = sqErrors.sum() / 2
        return J

    def batchGD(self, x, y, epochs):
        print("Training using batch gradient descent")
        epoch = 0
        outputChunk = int ( epochs / 10 )

        while epoch <= epochs:
            
            if epoch % outputChunk is 0:
                J = self.costFunction(x,y)
                print("Epoch=", epoch, "J=", J)
                
            partial_J_w_ih, partial_J_w_ho, partial_J_b_h, partial_J_b_o = \
                            self.deriv_costFunction( x, y )
            self.w_ih -= partial_J_w_ih
            self.w_ho -= partial_J_w_ho
            self.b_h -= partial_J_b_h
            self.b_o -= partial_J_b_o
            
            epoch += 1
            
    def miniBatchStochasticGD(self, x, y, batchSize, epochs):
        print("Training using stochastic minibatch gradient descent")
        epoch = 0
        fullSetSize = x.shape[0]
        outputChunk = int ( epochs / 10 )

        while epoch <= epochs:
            
            if epoch % outputChunk is 0:
                J = self.costFunction(x,y)
                print("Epoch=", epoch, "J=", J)
            
            counter = 0
            shuffled_x, shuffled_y = self.shuffleData(x, y)
            
            while counter < fullSetSize:
                x_batch, y_batch = self.getNextBatch(shuffled_x, shuffled_y, \
                                                     batchSize, counter)
                partial_J_w_ih, partial_J_w_ho, partial_J_b_h, partial_J_b_o = \
                            self.deriv_costFunction( x_batch, y_batch )
                self.w_ih -= partial_J_w_ih
                self.w_ho -= partial_J_w_ho
                self.b_h -= partial_J_b_h
                self.b_o -= partial_J_b_o
                counter += batchSize

            epoch += 1
                
    def getNextBatch(self, shuffled_x, shuffled_y, batchSize, counter):
            return shuffled_x[ counter: counter + batchSize], \
            shuffled_y[ counter: counter + batchSize]
                    
    def shuffleData(self, x, y):
        order = np.arange( x.shape[0] )
        random.shuffle( order )
        return x[order], y[order]

    def deriv_costFunction(self, x, y):
        self.yEst = self.forward_propagate(x)

        delta_o = np.multiply( ( self.yEst - y ), self.deriv_sigmoid(self.z_o) )
        partial_J_w_ho = np.dot( self.a_h.T, delta_o )

        ones_o = np.ones( delta_o.shape[0] )
        partial_J_b_o = np.dot( ones_o, delta_o  )

        delta_h = np.dot( delta_o, self.w_ho.T ) * self.deriv_sigmoid( self.z_h )
        partial_J_w_ih = np.dot( x.T, delta_h )
        
        ones_h = np.ones( delta_h.shape[0] )
        partial_J_b_h = np.dot( ones_h, delta_h)

        return partial_J_w_ih, partial_J_w_ho, partial_J_b_h, partial_J_b_o

    #Ancilliary functions to test analytic gradients
    def getParamsToVector(self):
        #vectorise and concat weights arrays
        weights = np.concatenate( ( self.w_ih.flatten(), self.w_ho.flatten() ) )
        # concat biases vectors
        biases = np.concatenate( ( self.b_h, self.b_o ) )
        # concat weights and biases into params
        params = np.concatenate( ( weights, biases ) )
        return params
    
    def setParamsFromVector(self, params):
        #starting point of w_ih weights in vectorised params
        w_ih_start_pos = 0
        #end point of w_ih weights in vectorised params
        w_ih_end_pos = self.hiddenLayerSize * self.inputLayerSize

        self.w_ih = np.reshape( params[ w_ih_start_pos : w_ih_end_pos ], \
                             ( self.inputLayerSize, self.hiddenLayerSize ) )

        #end point of w_ho weights in vectorised params
        w_ho_end_pos = w_ih_end_pos + self.hiddenLayerSize * self.outputLayerSize

        self.w_ho = np.reshape( params[ w_ih_end_pos : w_ho_end_pos ], \
                             ( self.hiddenLayerSize, self.outputLayerSize))

        #end point of b_h biases in vectorised params
        b_h_end_pos = w_ho_end_pos + self.hiddenLayerSize
        
        self.b_h = params[ w_ho_end_pos : b_h_end_pos ]
        
        #end point of b_o biases in vectorised params
        b_o_end_pos = b_h_end_pos + self.outputLayerSize
        
        self.b_o = params[ b_h_end_pos : b_o_end_pos ]


    def getGradientsToVector(self, x, y):
        partial_J_w_ih, partial_J_w_ho, partial_J_b_h, partial_J_b_o = \
                            self.deriv_costFunction( x, y )
        #vectorise gradients
        return np.concatenate( ( partial_J_w_ih.flatten(), partial_J_w_ho.flatten(), \
                                partial_J_b_h, partial_J_b_o ) )
    
    def getNumericalGradientsToVector(self, x, y):
        initialParams = self.getParamsToVector()
        numGrad = np.zeros( initialParams.shape )
        shiftVec = np.zeros( initialParams.shape )
        epsilon = 1e-4
        
        for param in range( len( initialParams ) ):
            shiftVec[ param ] = epsilon
            self.setParamsFromVector( initialParams + shiftVec )
            costFunction1 = self.costFunction( x,y )
            self.setParamsFromVector( initialParams - shiftVec )
            costFunction2 = self.costFunction( x,y )
    
            numGrad[ param ] = ( costFunction1 - costFunction2 ) / ( 2 * epsilon )
            shiftVec[ param ] = 0
            
        self.setParamsFromVector( initialParams )
        
        return numGrad
    


    