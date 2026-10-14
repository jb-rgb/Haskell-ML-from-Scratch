module Main where

import MathCore (euclideanDistance)

main :: IO ()
main = do
    putStrLn "Initializing ML Engine..."
    let dist = euclideanDistance [1.0, 2.0] [4.0, 6.0]
    putStrLn $ "Test distance: " ++ show dist