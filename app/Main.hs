module Main where

import DataLoader (loadMNIST)
import MathCore (Vector, Matrix, Sample)
import NeuralNet (NeuralNetwork(..), predict, trainStep)
import KNN (knnClassify)
import Control.Monad (foldM) -- Permite bucles con efectos secundarios (imprimir en pantalla)

-- 1. Generador Pseudoaleatorio Simple (Corregido el warning de 'tail')
pseudoRandomList :: Int -> Double -> [Double]
pseudoRandomList n seed = 
    let a = 1664525
        c = 1013904223
        m = 2^32
        nextSeed s = (a * s + c) `mod` m
        seeds = iterate nextSeed (round seed)
    in take n [(fromIntegral s / fromIntegral m) * 0.2 - 0.1 | s <- drop 1 seeds]

initRandomMatrix :: Int -> Int -> Double -> Matrix
initRandomMatrix rows cols seed = 
    let flat = pseudoRandomList (rows * cols) seed
        chunk [] _ = []
        chunk xs c = take c xs : chunk (drop c xs) c
    in chunk flat cols

-- 2. Motor de Evaluación
evaluateAccuracy :: NeuralNetwork -> [Sample] -> Double
evaluateAccuracy net samples =
    let correct = length [ () | (label, vec) <- samples, predict net vec == label ]
    in fromIntegral correct / fromIntegral (length samples) * 100.0

-- 3. Motor de Entrenamiento Visual (Monádico)
-- Usamos foldM para imprimir el progreso en la consola sin romper la pureza de la red.
trainEpochVerbose :: Double -> Int -> NeuralNetwork -> [Sample] -> IO NeuralNetwork
trainEpochVerbose lr epoch net samples = do
    putStrLn $ "\n--- Iniciando Época " ++ show epoch ++ " ---"
    let total = length samples
    
    -- Función paso a paso que actualiza la red e imprime cada 5,000 imágenes
    let step (currentNet, count) sample = do
            let nextNet = trainStep lr currentNet sample
            let nextCount = count + 1
            
            if nextCount `mod` 5000 == 0
                then putStrLn $ "  [Progreso] Procesadas " ++ show nextCount ++ " de " ++ show total ++ " imágenes..."
                else return ()
                
            return (nextNet, nextCount)
            
    -- Ejecuta el ciclo y extrae solo la red final
    (finalNet, _) <- foldM step (net, 0) samples
    return finalNet

-- 4. Punto de Entrada
main :: IO ()
main = do
    putStrLn "Cargando el dataset MNIST completo (60,000 imágenes)..."
    dataset <- loadMNIST "data/mnist_train.csv"
    
    -- Separamos: 50,000 para entrenar, 10,000 para validar
    let trainSet = take 50000 dataset
        testSet  = drop 50000 dataset
        
    putStrLn "\n--- Inicializando Red Neuronal (128 neuronas ocultas) ---"
    let w1 = initRandomMatrix 128 784 42.0
        b1 = replicate 128 0.0
        w2 = initRandomMatrix 10 128 123.0
        b2 = replicate 10 0.0
        net0 = NeuralNetwork w1 b1 w2 b2
        
    putStrLn $ "Precisión de la red ANTES de entrenar: " ++ show (evaluateAccuracy net0 testSet) ++ "%"

    -- Entrenamiento de 2 épocas con retroalimentación visual en consola
    let lr = 0.01
    net1 <- trainEpochVerbose lr 1 net0 trainSet
    putStrLn $ "=> Precisión global tras Época 1: " ++ show (evaluateAccuracy net1 testSet) ++ "%"
    
    net2 <- trainEpochVerbose lr 2 net1 trainSet
    putStrLn $ "=> Precisión global tras Época 2: " ++ show (evaluateAccuracy net2 testSet) ++ "%"

    putStrLn "\n=================================================="
    putStrLn "   COMPARATIVA DE ARQUITECTURAS: k-NN vs Red Neuronal"
    putStrLn "=================================================="
    -- Tomamos solo 50 imágenes para la prueba cruzada. Si usaramos las 10,000, 
    -- el k-NN tardaría horas en calcular todas las distancias.
    let smallTestSet = take 50 testSet
    putStrLn "Evaluando una muestra de 50 imágenes nuevas..."
    
    -- Evaluar Red Neuronal
    putStrLn "\n1. Evaluando con Red Neuronal..."
    let nnCorrect = length [ () | (lbl, vec) <- smallTestSet, predict net2 vec == lbl ]
    putStrLn $ "  -> Aciertos NN: " ++ show nnCorrect ++ " de 50"
    putStrLn "  -> Tiempo de Inferencia NN: ¡Casi instantáneo! (2 multiplicaciones matriciales por imagen)"
    
    -- Evaluar k-NN
    putStrLn "\n2. Evaluando con k-NN (k=3)..."
    putStrLn "  (Nota: k-NN tardará unos segundos. Debe medir la distancia Euclidiana contra 50,000 imágenes para CADA predicción)"
    let knnCorrect = length [ () | (lbl, vec) <- smallTestSet, knnClassify 3 trainSet vec == lbl ]
    putStrLn $ "  -> Aciertos k-NN: " ++ show knnCorrect ++ " de 50"
    putStrLn "  -> Tiempo de Inferencia k-NN: Computacionalmente pesado."