import numpy as np

def findTrajectory(in_these_lines):
    i=0
    for line in in_these_lines:
        cols = line.split(",")
        if cols[0] != '1' and cols[1] != '1':
            i = i+1
        else:
            break
    trajectory = np.array(in_these_lines[0:i+1])
    return trajectory


def parse(file_name):
    inputfile = open(file_name)
    these_lines = inputfile.read().splitlines()
    i=1
    xlist=list()
    ylist=list()
    thetalist=list()
    thetadotlist=list()
    actionlist=list()
    rewardlist=list()
    while i<len(these_lines):
        lines_wout_header = these_lines[i:]
        traj = np.array(findTrajectory(lines_wout_header))
        nStep = len(traj)
        xep = np.empty(nStep)
        yep = np.empty(nStep)
        thetaep = np.empty(nStep)
        thetadotep = np.empty(nStep)
        actionep = np.empty(nStep-1)
        rewardep = np.empty(nStep-1)
        index = 0
        for step in traj:
            rows = step.split(",")
            xep[index] = rows[2]
            yep[index] = rows[3]
            thetaep[index] = rows[4]
            thetadotep[index] = rows[5]
            if index<nStep-1:
                actionep[index] = rows[6]
                rewardep[index] = rows[7]
            index = index+1

        xlist.append(xep)
        ylist.append(yep)
        thetalist.append(thetaep)
        thetadotlist.append(thetadotep)
        actionlist.append(actionep)
        rewardlist.append(rewardep)

        i=i+nStep

    return (xlist, ylist, thetalist, thetadotlist, actionlist, rewardlist)

