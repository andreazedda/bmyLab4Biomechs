"""
    Description:
        This module is used to define phases (ascending, descending, stable) of repetions of a given recording.
        Recordings are signals that are periodic because they are Euler angles.
        Recordings are made by a sensor that is attached to a human body.
        Signals are loaded from a file.
        The recording file contains four columns: time, rotations on x axis, rotations on y axis, rotations on z axis.
        The period of repetions is calculated considering the rotation with the highest amplitude.
        Before calculating the period of repetions, the signal is filtered.
    
    Author:
        Andrea Zedda
"""
