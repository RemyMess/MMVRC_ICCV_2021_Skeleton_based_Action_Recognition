import numpy as np
from tqdm import tqdm

"""
A module containing different classes related to checking the distance between two frames, and repairing the data if it is too big.
Format of data: N = number of samples
                C = dimesion
                T = number of frames
                V = number of joints
                M = number of bodies (THE FILE EXPECTS M=2)
"""

def getInvalidFrames(data, thresholdOnePerson = 100, thresholdTwoPersons = 130):
    """
    Expects the Data as an numpy array of the form N, C, T, V, (M = 2)
    Returns a list of sample ids corresponding to samples, which move faster than their respective threshold
    """
    N,C,T,V,M = data.shape
    invalid_samples = []
    
    for sample in tqdm(range(N)):
        numBodies = numberBodies(data[sample,:,:,:,:])
        
        averageDist = averageDistance(data[sample,:,:,:,:])

        if (averageDist > thresholdOnePerson and numBodies == 1) or (averageDist > thresholdTwoPersons and numBodies == 2):
            invalid_samples.append(sample)
            
    return invalid_samples


def findEffectiveLength(sample):
    """
    Expects a sample of shape C,T,V,M, returns the number of frames, after which only 0-frames follow.
    """
    C,T,V,M = sample.shape
    non_zero_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    return T - non_zero_frames[::-1].argmax()


def numberBodies(sample):
    """
    Expects a sample of shape C,T,V,M, returns the number of bodies.
    """
    if sample[:,:,:,1].sum() == 0:
        return 1
    else:
        return 2

def averageDistance(sample):
    """
    Expects a sample in the format C,T,V,M, returns the average distance (2-distance) between 2 frames
    """
    effective_length = findEffectiveLength(sample)
    totalDist = 0
    for frame in range(effective_length-1):
        totalDist += distanceFrames(sample[:,frame,:,:], sample[:,frame+1,:,:])
    return totalDist/effective_length

def distanceFrames(frame1, frame2):
    """
    Expects 2 frames in the format C,V,M, returns the 2-distance between them
    """
    return np.linalg.norm(frame1-frame2)


def switchPeople(sample):
    """
    Expects a sample of the form C,T,V,M
    If there is only 1 body in the sample, it returns the sample.
    Otherwise, it goes through each frame and switches the bodies, if it reduces the distance to the previous frame. Returns the new sample.
    """
    if numberBodies(sample) == 1:
        return sample

    effective_length = findEffectiveLength(sample)
    for frame in range(effective_length-1):
        nextFrameSwitched = np.flip(sample[:,frame+1,:,:],axis = 2)
        if distanceFrames(nextFrameSwitched, sample[:,frame,:,:]) < distanceFrames(sample[:,frame+1,:,:], sample[:,frame,:,:]):
            sample[:,frame+1,:,:] = nextFrameSwitched

    return sample

def interpolate(sample, frame1, frame2):
    for i in range(frame1+1,frame2):
        alpha = (i-frame1)/(frame2-frame1)
#        print(frame1, frame2, i ,alpha)
        sample[:,i,:] = convexCombination(sample[:,frame1,:],sample[:,frame2,:],alpha)
    return sample

def convexCombination(x, y, alpha):
    return alpha*x + (1-alpha)*y

def eliminate_spikes(sample, threshold = 0.7, possibleFirstFrames = 15, thresholdRejection = 0.2):
    """
    Expects a sample of the form C,T,V
    searches frames whose distance is below the respective threshold and linearly interpolates between them
    tries first possibleFirstFrames as valid starting frames
    If this proceedure would delete more than thresholdRjection % of the frames, it returns the sample instead
    """
    C,T,V =sample.shape
    
    effective_length = findEffectiveLength(sample.reshape(C,T,V,1))

    #list of already tested frames, no need to run them twice
    tested = []

    #list of interpolated samples, as well as the number of deleted frames, indexed by the starting frame
    samples_interpolated = []
    list_deleted_frames = []
    
    for startFrame in range(min(possibleFirstFrames,effective_length-1)):
        if startFrame in tested:
            continue

        current_sample = sample.copy()

        #pad frames left of starting frame
        for i in range(startFrame):
            current_sample[:,i,:] = current_sample[:,startFrame,:]

        number_deleted_frames = startFrame

        #left index forinterpolation
        left = startFrame
        while left < effective_length-1:

            #right index for interpolation
            for right in range(left+1,effective_length):
                if distanceFrames(sample[:,left,:],sample[:,right,:]) <= threshold*(right-left):
                    if right-left > 1:
                        current_sample = interpolate(current_sample, left,right)
                        number_deleted_frames += right-left-1
                    left = right
                    tested.append(left)
                    break
                if right == effective_length-1:
                    current_sample[:,left+1:,] = 0
                    number_deleted_frames += right-left
                    left = right

        samples_interpolated.append(current_sample)
        list_deleted_frames.append(number_deleted_frames)

    if len(list_deleted_frames) == 0:
        return sample
    
    arr_deleted_frames = np.array(list_deleted_frames)
    best_start = np.argmin(arr_deleted_frames)


    if arr_deleted_frames[best_start] < thresholdRejection*effective_length:
        return samples_interpolated[best_start]
        
    else:
        return sample
        
                
    

    
