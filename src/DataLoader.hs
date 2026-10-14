module DataLoader (
    loadMNIST
) where

import MathCore (Sample, Label, Vector)

parseLine :: String -> Sample
parseLine line = 
    let cleanLine = map (\c -> if c == ',' then ' ' else c) line
        values = map read (words cleanLine) :: [Double]
    in case values of
        (l:pxs) -> (round l, map (/ 255.0) pxs)
        []      -> error "Línea vacía o formato incorrecto en CSV"

loadMNIST :: FilePath -> IO [Sample]
loadMNIST path = do
    content <- readFile path
    let allLines = lines content
        dataLines = case allLines of
            (first:rest) | take 1 first == "l" -> rest 
            _ -> allLines
    return (map parseLine dataLines)