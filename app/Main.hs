module Main where

import DataLoader (loadMNIST)
import KNN (knnClassify)
import MathCore (Vector, Matrix)
import NeuralNet (NeuralNetwork(..), predict)

-- Funciones auxiliares para crear matrices y vectores de prueba
dummyVector :: Int -> Double -> Vector
dummyVector size val = replicate size val

dummyMatrix :: Int -> Int -> Double -> Matrix
dummyMatrix rows cols val = replicate rows (replicate cols val)

main :: IO ()
main = do
    putStrLn "Cargando el dataset MNIST..."
    dataset <- loadMNIST "data/mnist_train.csv"
    
    let trainSubset = take 1000 dataset
        (targetLabel, targetVector) = dataset !! 1000
    
    putStrLn "\n--- Prueba 1: Modelo k-NN ---"
    let knnPrediction = knnClassify 3 trainSubset targetVector
    putStrLn $ "Predicción k-NN: " ++ show knnPrediction
    putStrLn $ "Etiqueta real: " ++ show targetLabel

    putStrLn "\n--- Prueba 2: Red Neuronal (Pipeline) ---"
    -- Inicializamos una red con 128 neuronas ocultas y 10 neuronas de salida
    let dummyNet = NeuralNetwork 
            (dummyMatrix 128 784 0.01) -- w1: 128x784
            (dummyVector 128 0.0)      -- b1: 128
            (dummyMatrix 10 128 0.01)  -- w2: 10x128
            (dummyVector 10 0.0)       -- b2: 10
            
    -- Pasamos la imagen por la red
    let nnPrediction = predict dummyNet targetVector
    
    putStrLn "El Forward Pass completó exitosamente."
    putStrLn $ "Predicción de la red sin entrenar: " ++ show nnPrediction