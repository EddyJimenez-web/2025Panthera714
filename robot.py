#!/usr/bin/env python3
#
# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#

import typing
import wpilib
import commands2

from robotcontainer import RobotContainer


class MyRobot(commands2.TimedCommandRobot):
    """
    Command v2 robots are encouraged to inherit from TimedCommandRobot, which
    has an implementation of robotPeriodic which runs the scheduler for you
    """

    autonomousCommand: typing.Optional[commands2.Command] = None
    testCommand: typing.Optional[commands2.Command] = None

    def robotInit(self) -> None:
        """
        This function is run when the robot is first started up and should be used for any
        initialization code.
        """

        # Instantiate our RobotContainer.  This will perform all our button bindings, and put our
        # autonomous chooser on the dashboard.
        self.robotContainer = RobotContainer(self)

    def disabledInit(self) -> None:
        """This function is called once each time the robot enters Disabled mode."""
        self.robotContainer.disabledInit()

    def disabledPeriodic(self) -> None:
        """This function is called periodically when disabled"""

    def autonomousInit(self) -> None:
        """This autonomous runs the autonomous command selected by your RobotContainer class."""
        self.robotContainer.autonomousInit()
        self.autonomousCommand = self.robotContainer.getAutonomousCommand()

        # schedule the autonomous command (example)
        if self.autonomousCommand:
            self.autonomousCommand.schedule()

    def autonomousPeriodic(self) -> None:
        """This function is called periodically during autonomous"""

    def teleopInit(self) -> None:
        # This makes sure that the autonomous stops running when
        # teleop starts running. If you want the autonomous to
        # continue until interrupted by another command, remove
        # this line or comment it out.
        if self.autonomousCommand:
            self.autonomousCommand.cancel()
        self.robotContainer.teleopInit()

    def teleopPeriodic(self) -> None:
        """This function is called periodically during operator control"""

    def testInit(self) -> None:
        # Cancels all running subsystems at the start of test mode
        commands2.CommandScheduler.getInstance().cancelAll()
        self.testCommand = self.robotContainer.getTestCommand()

        # schedule the autonomous command (example)
        if self.testCommand is not None:
            self.testCommand.schedule()

    def testPeriodic(self) -> None:
        """This function is called periodically during test mode"""