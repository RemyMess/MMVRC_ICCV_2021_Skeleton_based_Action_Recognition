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
    Expects the Data as an numpy array of the form N, M, T, V, C
    Returns a list of sample ids corresponding to samples, which move faster than their respective threshold
    """
    N,M,T,V,C = data.shape
    invalid_samples = []
    
    for sample in tqdm(range(N)):
        numBodies = numberBodies(data[sample,:,:,:,:])
        
        averageDist = averageDistance(data[sample,:,:,:,:])

        if (averageDist > thresholdOnePerson and numBodies == 1) or (averageDist > thresholdTwoPersons and numBodies == 2):
            invalid_samples.append(sample)
            
    return invalid_samples


def findEffectiveLength(sample):
    """
    Expects a sample of shape M,T,V,C returns the number of frames, after which only 0-frames follow.
    """
    M,T,V,C = sample.shape
    non_zero_frames = (sample != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    return T - non_zero_frames[::-1].argmax()


def numberBodies(sample):
    """
    Expects a sample of shape M,T,V,C returns the number of bodies.
    """
    if sample[1,:,:,:].sum() == 0:
        return 1
    else:
        return 2

def averageDistance(sample):
    """
    Expects a sample in the format M,T,V,C, returns the average distance (2-distance) between 2 frames
    """
    effective_length = findEffectiveLength(sample)
    totalDist = 0
    for frame in range(effective_length-1):
        totalDist += distanceFrames(sample[:,frame,:,:], sample[:,frame+1,:,:])
    return totalDist/effective_length

def distanceFrames(frame1, frame2):
    """
    Expects 2 frames in any format, returns the 2-distance between them
    """
    return np.linalg.norm(frame1-frame2)

def frameIsNull(frame):
    return frame.sum() == 0

def findNonZeroFrames(sample):
    T, V, C = sample.shape

    nonZeroFrames = []
    for idx in range(T):
        if not frameIsNull(sample[idx, :, :]):
            nonZeroFrames.append(idx)
    return nonZeroFrames

def switchPeople(sample):
    """
    Expects a sample of the form M,T,V,C
    If there is only 1 body in the sample, it returns the sample.
    Otherwise, it goes through each frame and switches the bodies, if it reduces the distance to the previous frame. Returns the new sample.
    """
    if numberBodies(sample) == 1:
        return sample

    M,T,V,C = sample.shape
    effective_length = findEffectiveLength(sample)

    currentSample = sample.copy()

    hasTwoBodies = []
    for idx in range(effective_length):
        if currentSample[0,idx,:,:].sum() != 0 and currentSample[1,idx,:,:].sum() != 0:
            hasTwoBodies.append(idx)

    # If there is only one body, move it to index 0
    if len(hasTwoBodies) == 0:
        for frame in range(effective_length):
            frame = currentSample[:,frame,:,:]
            if frame[1,:,:].sum != 0:
                currentSample[:,frame,:,:] = np.flip(frame, axis=0)
        return currentSample

    person0 = currentSample[0,hasTwoBodies[0],:,:]
    person1 = currentSample[1,hasTwoBodies[0],:,:]

    #find correct person in frames before first appearance of two bodies
    for frame in reversed(range(hasTwoBodies[0])):
        if frameIsNull(currentSample[:,frame,:,:]):             #don't touch null frames
            continue
        if not frameIsNull(currentSample[0,frame,:,:]):         #find person in current frame
            positionPerson = 0
            person = currentSample[0,frame,:,:]
        else:
            positionPerson = 1
            person = currentSample[1, frame, :, :]

        if distanceFrames(person,person0) < distanceFrames(person,person1):     #find Position person should have
            positionShouldBe = 0
            person0 = person
        else:
            positionShouldBe = 1
            person1 = person

        if positionShouldBe != positionPerson:                                  #if necessary: switch frame
            currentSample[:,frame,:,:] = np.flip(currentSample[:,frame,:,:],axis = 0)

    #reset the bodies
    person0 = currentSample[0, hasTwoBodies[0], :, :]
    person1 = currentSample[1, hasTwoBodies[0], :, :]

    #correct position in all following frames
    for frame in range(hasTwoBodies[0]+1,effective_length):
        if frame in hasTwoBodies:                               #if frame has two bodies, buisness as usual
            frameFlipped = np.flip(currentSample[:,frame,:,:], axis = 0)
            bodies = np.array([person0,person1])

            if distanceFrames(bodies,frameFlipped) < distanceFrames(bodies,currentSample[:,frame,:,:]):
                currentSample[:,frame,:,:] = frameFlipped

            person0 = currentSample[0,frame,:,:]
            person1 = currentSample[1,frame,:,:]
            continue

        else:                                                   #same as above
            if frameIsNull(currentSample[:, frame, :, :]):      #don't touch null frames
                continue
            if not frameIsNull(currentSample[0, frame, :, :]):  # find person in current frame
                positionPerson = 0
                person = currentSample[0, frame, :, :]
            else:
                positionPerson = 1
                person = currentSample[1, frame, :, :]

            if distanceFrames(person, person0) < distanceFrames(person, person1):  # find Position person should have
                positionShouldBe = 0
                person0 = person
            else:
                positionShouldBe = 1
                person1 = person

            if positionShouldBe != positionPerson:  # if necessary: switch frame
                currentSample[:, frame, :, :] = np.flip(currentSample[:, frame, :, :], axis=0)

    return currentSample

def interpolate(sample, frame1, frame2):
    """expects sample in the format T,V,C"""
    for i in range(frame1+1,frame2):
        alpha = (i-frame1)/(frame2-frame1)
        #print(frame1, frame2, i ,alpha)
        sample[i,:,:] = convexCombination(sample[frame1,:,:],sample[frame2,:,:],alpha)
        sample[i,:,2] = 0
    return sample

def convexCombination(x, y, alpha):
    return alpha*y + (1-alpha)*x

def eliminate_spikes(sample, threshold = 0.7, possibleFirstFrames = 15, thresholdRejection = 0.2):
    """
    Expects a sample of the form T,V,C
    searches frames whose distance is below the respective threshold and linearly interpolates between them
    tries first possibleFirstFrames as valid starting frames
    If this proceedure would delete more than thresholdRjection % of the frames, it returns the sample instead
    """
    T,V,C =sample.shape

    nonZeroFrames = findNonZeroFrames(sample)

    #list of already tested frames, no need to run them twice
    tested = []

    #list of interpolated samples, as well as the number of deleted frames, indexed by the starting frame
    samples_interpolated = []
    list_deleted_frames = []
    
    for startIdx in range(min(possibleFirstFrames,len(nonZeroFrames)-1)):
        startFrame = nonZeroFrames[startIdx]
        if startFrame in tested:
            continue

        current_sample = sample.copy()
        current_sample[:,:,2] = 1
        number_deleted_frames = 0

        #pad frames left of starting frame
        for i in range(startFrame):
            current_sample[i,:,:] = current_sample[startFrame,:,:]
            current_sample[i,:,2] = 0                               #deleted frame becomes invisible
            if i in nonZeroFrames:
                number_deleted_frames += 1

        #left index forinterpolation
        leftIdx = startIdx
        while leftIdx < len(nonZeroFrames)-1:

            #right index for interpolation
            for rightIdx in range(leftIdx+1,len(nonZeroFrames)):
                left = nonZeroFrames[leftIdx]
                right = nonZeroFrames[rightIdx]

                if distanceFrames(sample[left,:,:],sample[right,:,:]) <= threshold*(right-left):
                    if right-left > 1:
                        current_sample = interpolate(current_sample, left, right)
                        for idx in range(left+1,right):
                            if idx in nonZeroFrames:
                                number_deleted_frames += 1
                    leftIdx = rightIdx
                    tested.append(leftIdx)
                    break

                if rightIdx == len(nonZeroFrames)-1:
                    for idx in range(left+1,right+1):
                        current_sample[idx,:,:2] = current_sample[left,:,:2]
                        current_sample[idx,:,2] = 0
                        if idx in nonZeroFrames:
                            number_deleted_frames += 1
                    leftIdx = rightIdx

        samples_interpolated.append(current_sample)
        list_deleted_frames.append(number_deleted_frames)

    if len(list_deleted_frames) == 0:
        return sample
    
    arr_deleted_frames = np.array(list_deleted_frames)
    best_start = np.argmin(arr_deleted_frames)

    if arr_deleted_frames[best_start] < thresholdRejection*len(nonZeroFrames):
        sample = samples_interpolated[best_start]
        for idx in range(nonZeroFrames[-1] + 1, T):
            #remove null-frames at the end
            sample[idx, :, :2] = sample[nonZeroFrames[-1], :, :2]
            sample[idx, :, 2] = 0
        return sample
        
    else:
        return padNullFrames(sample)

def padNullFrames(sample):
    T, V, C = sample.shape
    nonZeroFrames = findNonZeroFrames(sample)

    sample[:,:,2] = 1

    for idx in range(nonZeroFrames[0]):
        sample[idx,:,:2] = sample[nonZeroFrames[0],:,:2]
        sample[idx,:,2] = 0

    for nonNullIdx in range(len(nonZeroFrames)-1):
        sample = interpolate(sample, nonZeroFrames[nonNullIdx], nonZeroFrames[nonNullIdx+1])

    for idx in range(nonZeroFrames[-1]+1,T):
        sample[idx,:,:2] = sample[nonZeroFrames[-1],:,:2]
        sample[idx,:,2] = 0

    return sample





        
                
    

    
