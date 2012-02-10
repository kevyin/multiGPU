{-# LANGUAGE ParallelListComp, CPP, ForeignFunctionInterface #-}
--------------------------------------------------------------------------------
--
-- Module    : MultiGPU
-- Copyright : (c) 2012 Kevin Ying
-- License   : BSD
--
--
--------------------------------------------------------------------------------

module Main where

--import Data.Array.Unboxed
--import Graphics.Pgm
import Foreign
import Foreign.CUDA             (DevicePtr, withDevicePtr) 
import Foreign.CUDA.Runtime.Stream

reduceKernel :: DevicePtr Float -> DevicePtr Float -> Int -> Int -> Int -> Int -> Stream -> IO ()
reduceKernel a1 a2 a3 a4 a5 a6 a7 =
  withDevicePtr a1 $ \a1' ->
  withDevicePtr a2 $ \a2' ->
  reduceKernel'_ a1' a2' a3 a4 a5 a6 a7

foreign import ccall unsafe "simpleMultiGPU.h launch_reduceKernel"
  reduceKernel'_ :: Ptr Float -> Ptr Float -> Int -> Int -> Int -> Int -> Stream -> IO ()

main :: IO ()
main = do
  putStrLn "WEFWEF"
