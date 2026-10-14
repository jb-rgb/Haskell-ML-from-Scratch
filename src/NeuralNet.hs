module NeuralNet(
    NeuralNetwork(..),
    softmax,
    fordwardPass,
    predict
) where

import MathCore (Vector, Matrix, Label, matrixVectorMultiply, addVectors, reluVector)
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

fordwardPass :: NeuralNetwork -> Vector -> Vector
fordwardPass (NeuralNetwork w1 b1 w2 b2) input =
    let
        z1 = addVectors (matrixVectorMultiply w1 input) b1
        a1 = reluVector z1
        z2 = addVectors (matrixVectorMultiply w2 a1) b2
        a2 = softmax z2
    in a2

predict :: NeuralNetwork -> Vector -> Label
predict nn input =
    let probabilities = fordwardPass nn input
        maxProb = maximum probabilities
    in case elemIndex maxProb probabilities of
        Just idx -> idx
        Nothing  -> error "Prediction failed: empty probability vector"