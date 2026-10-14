module Main where

import DataLoader (loadMNIST)
import KNN (knnClassify)

main :: IO ()
main = do
    putStrLn "Cargando el dataset MNIST..."
    dataset <- loadMNIST "data/mnist_train.csv"
    
    let trainSubset = take 1000 dataset
        (targetLabel, targetVector) = dataset !! 1000

    putStrLn "Iniciando clasificación k-NN (k=3)..."

    let prediction = knnClassify 3 trainSubset targetVector

    putStrLn $ "Predicción del modelo: " ++ show prediction
    putStrLn $ "Etiqueta real: " ++ show targetLabel
    
    if prediction == targetLabel
        then putStrLn "¡El modelo acertó!"
        else putStrLn "El modelo falló esta vez."    