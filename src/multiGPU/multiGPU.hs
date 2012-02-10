{-# LANGUAGE ParallelListComp, CPP, BangPatterns, ForeignFunctionInterface #-}
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
import System.Random.MWC
import Data.List
import Control.Monad
import Foreign
import Foreign.CUDA             (HostPtr, DevicePtr, withDevicePtr, withHostPtr) 
import Foreign.CUDA.Runtime.Stream      as CR
import qualified Foreign.CUDA.Runtime   as CR

import qualified Data.Vector.Unboxed    as U


-- plan
data Plan = Plan
  {
    device            :: Int,
    dataN             :: Int,

    h_data            :: HostPtr Float,
    h_sum             :: HostPtr Float,

    d_data            :: DevicePtr Float,
    d_sum             :: DevicePtr Float,

    h_Sum_from_device :: HostPtr Float,
    stream            :: Stream
  }
  deriving Show


reduceKernel :: DevicePtr Float -> DevicePtr Float -> Int -> Int -> Int -> Int -> Stream -> IO ()
reduceKernel a1 a2 a3 a4 a5 a6 a7 =
  withDevicePtr a1 $ \a1' ->
  withDevicePtr a2 $ \a2' ->
  reduceKernel'_ a1' a2' a3 a4 a5 a6 a7

foreign import ccall unsafe "simpleMultiGPU.h launch_reduceKernel"
  reduceKernel'_ :: Ptr Float -> Ptr Float -> Int -> Int -> Int -> Int -> Stream -> IO ()

main :: IO ()
main = let max_gpu_count = 32
       in do
  gpu_n' <- CR.count
  putStrLn $ show $ "Number of GPUs found: " ++ show gpu_n'
  let gpu_n = min max_gpu_count gpu_n'

  withPlans gpu_n $ \plans -> do
    forM_ plans $ \plan -> do
      CR.set (device plan)


  putStrLn "Finished"
  


withPlans :: Int -> ([Plan] -> IO ()) ->  IO ()
withPlans gpu_n action = 
  let data_n   = 1048576 * 32
      block_n  = 32
      thread_n = 256
      accum_n  = block_n * thread_n
      devices  = [0..(gpu_n-1)]
  in do
  plans <- forM devices $ \i -> do
    let dataN = if (i < data_n `mod` gpu_n) then (data_n `div` gpu_n) + 1 else data_n `div` gpu_n
    CR.set i
    stream <- CR.create
    -- Allocate memory
    -- Host
    --seed <- newStdGen
    --let h_data' = take dataN $ randoms seed
    h_data' <- randomList dataN 
    h_data <- CR.mallocHostArray [] dataN 
    withHostPtr h_data $ \ptr -> pokeArray ptr h_data'

    h_sum <- CR.mallocHostArray [] gpu_n
    h_Sum_from_device <- CR.mallocHostArray [] accum_n 

    -- Device
    d_data <- CR.mallocArray (dataN   * sizeOf (undefined :: DevicePtr Float))
    d_sum  <- CR.mallocArray (accum_n * sizeOf (undefined :: DevicePtr Float))


    return $ Plan i dataN h_data h_sum d_data d_sum h_Sum_from_device stream

  action plans

  -- clean up
  forM_ plans $ \plan -> do
    CR.set (device plan)
    CR.freeHost (h_data plan)
    CR.freeHost (h_sum  plan)
    CR.freeHost (h_Sum_from_device plan)

    CR.free (d_data plan)
    CR.free (d_sum plan)
    CR.destroy (stream plan)

randomList :: Int -> IO [Float]
randomList n = withSystemRandom $ \gen -> return . U.toList =<< U.replicateM n (uniform gen :: IO Float)

