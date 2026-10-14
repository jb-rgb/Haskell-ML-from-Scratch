module Main where

import DataLoader (loadMNIST)

main :: IO ()
main = do
    putStrLn "Cargando el dataset MNIST..."
    dataset <- loadMNIST "data/mnist_train.csv"
    
    putStrLn $ "Total de imágenes cargadas: " ++ show (length dataset)
    
    case dataset of
        ((primeraEtiqueta, primerosPixeles):_) -> do
            putStrLn $ "Etiqueta de la primera imagen: " ++ show primeraEtiqueta
            putStrLn $ "Dimensiones del vector de la primera imagen: " ++ show (length primerosPixeles)
        [] -> putStrLn "Error: El dataset está vacío."