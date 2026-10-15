module NeuralNet(
    NeuralNetwork(..),
    softmax,
    forwardPass,
    predict,
    trainStep
) where

import MathCore (Vector, Matrix, Label, Sample, matrixVectorMultiply, addVectors, subtractVectors, reluVector, oneHotEncode, transposeMatrix, reluDerivativeVector, scalarMultiply, outerProduct)
import Data.List (elemIndex)

data NeuralNetwork = NeuralNetwork {
    w1 :: Matrix,
    b1 :: Vector,
    w2 :: Matrix,
    b2 :: Vector
} deriving (Show)

softmax :: Vector -> Vector
softmax v =
    let maxVal = maximum v
        exps = map (\x -> exp (x - maxVal)) v
        sumExps = sum exps
    in map (/ sumExps) exps

forwardPass :: NeuralNetwork -> Vector -> Vector
forwardPass (NeuralNetwork w1 b1 w2 b2) input =
    let
        z1 = addVectors (matrixVectorMultiply w1 input) b1
        a1 = reluVector z1
        z2 = addVectors (matrixVectorMultiply w2 a1) b2
        a2 = softmax z2
    in a2

predict :: NeuralNetwork -> Vector -> Label
predict nn input =
    let probabilities = forwardPass nn input
        maxProb = maximum probabilities
    in case elemIndex maxProb probabilities of
        Just idx -> idx
        Nothing  -> error "Prediction failed: empty probability vector"

trainStep :: Double -> NeuralNetwork -> Sample -> NeuralNetwork
trainStep lr (NeuralNetwork w1 b1 w2 b2) (label, x) =
    let
        z1 = addVectors (matrixVectorMultiply w1 x) b1
        a1 = reluVector z1
        
        z2 = addVectors (matrixVectorMultiply w2 a1) b2
        a2 = softmax z2
        
        y = oneHotEncode 10 label
        dz2 = subtractVectors a2 y
        
        dz1 = zipWith (*) (matrixVectorMultiply (transposeMatrix w2) dz2) (reluDerivativeVector z1)
        
        dw2 = outerProduct dz2 a1
        db2 = dz2
        dw1 = outerProduct dz1 x
        db1 = dz1
        
        updateMatrix m dm = zipWith (\row drow -> zipWith (-) row (scalarMultiply lr drow)) m dm
        updateVector v dv = zipWith (-) v (scalarMultiply lr dv)
        
        newW1 = updateMatrix w1 dw1
        newB1 = updateVector b1 db1
        newW2 = updateMatrix w2 dw2
        newB2 = updateVector b2 db2
        
    in NeuralNetwork newW1 newB1 newW2 newB2