from __future__ import annotations
import commands2

from wpimath.geometry import Rotation2d, Pose2d, Translation2d


class ResetXY(commands2.Command):
    def __init__(self, x, y, headingDegrees, drivetrain, resetGyro=True):
        """
        Reset the starting (X, Y) and heading (in degrees) of the robot to where they should be.
        :param x: X
        :param y: X
        :param headingDegrees: heading (for example: 0 = "North" of the field, 180 = "South" of the field)
        :param drivetrain: drivetrain on which the (X, Y, heading) should be set
        """
        super().__init__()

        self.drivetrain = drivetrain
        self.addRequirements(drivetrain)

        self.resetGyro = resetGyro
        self.headingDegrees = headingDegrees
        self.x = x
        self.y = y

    def initialize(self):
        heading = self.headingDegrees
        if heading is None:
            heading = self.drivetrain.getPose().rotation().degrees()
        elif callable(heading):
            heading = heading()
        position = Pose2d(Translation2d(self.x, self.y), Rotation2d.fromDegrees(heading))
        self.drivetrain.resetOdometry(position, resetGyro=self.resetGyro)

    def isFinished(self) -> bool:
        return True  # this is an instant command, it finishes right after it initialized

    def execute(self):
        """
        nothing to do here, this is an instant command
        """

    def end(self, interrupted: bool):
        """
        nothing to do here, this is an instant command
        """


class ResetSwerveFront(commands2.Command):
    def __init__(self, drivetrain):
        super().__init__()
        self.drivetrain = drivetrain
        self.addRequirements(drivetrain)

    def initialize(self):
        pose = self.drivetrain.getPose()
        self.drivetrain.resetOdometry(pose)

    def isFinished(self) -> bool:
        return True  # this is an instant command, it finishes right after it initialized

    def execute(self):
        """
        nothing to do here, this is an instant command
        """

    def end(self, interrupted: bool):
        """
        nothing to do here, this is an instant command
        """

    