#
# Copyright (c) FIRST and other WPILib contributors.
# Open Source Software; you can modify and/or share it under the terms of
# the WPILib BSD license file in the root directory of this project.
#

from __future__ import annotations
import math

import commands2
import rev
import wpimath
import wpilib
import typing

from commands2 import cmd, InstantCommand, RunCommand
from commands2.button import CommandGenericHID
from wpilib import XboxController
from wpimath.geometry import Pose2d, Rotation2d, Translation2d
import constants
from commands.elevatorcommands import MoveElevatorAndArm

from commands.jerky_trajectory import JerkyTrajectory, SwerveTrajectory
from commands.swervetopoint import SwerveToSide
from constants import DriveConstants, OIConstants
from subsystems.drivesubsystem import DriveSubsystem, BadSimPhysics
from subsystems.arm import Arm, ArmConstants

from commands.gotopoint import GoToPoint
from commands.reset_xy import ResetXY, ResetSwerveFront

from autofactory import AutoFactory


class RobotContainer:
    """
    This class is where the bulk of the robot should be declared. Since Command-based is a
    "declarative" paradigm, very little robot logic should actually be handled in the :class:`.Robot`
    periodic methods (other than the scheduler calls). Instead, the structure of the robot (including
    subsystems, commands, and button mappings) should be declared here.
    """

    def __init__(self, robot) -> None:
        wpilib.DriverStation.silenceJoystickConnectionWarning(True)
        self.robot: MyRobot = robot

        # The driver's controller
        self.driverController = CommandGenericHID(0)
        self.scoringController = CommandGenericHID(1)
        self.trajectoryBoard = CommandGenericHID(2)
        self.trajectorySide = "left"  # some kind of initial value
        self.trajectoryLetter = "A"  # we need some kind of initial value

        self.arm = Arm(leadMotorCANId=DriveConstants.kArmLeadMotorCanId, followMotorCANId=None)

        # The robot's subsystems
        from subsystems.limelight_camera import LimelightCamera
        from subsystems.photon_tag_camera import PhotonTagCamera
        self.frontRightCamera = LimelightCamera("limelight-aiming")  # name of your camera goes in parentheses
        self.frontLeftCamera = PhotonTagCamera("Arducam_Front")
        self.rearCamera = PhotonTagCamera("Arducam_Rear")

        from subsystems.intake import Intake
        from playingwithfusion import TimeOfFlight

        self.rangefinder = TimeOfFlight(DriveConstants.kIntakeRangefinderCanId)
        self.rangefinder.setRangingMode(TimeOfFlight.RangingMode.kShort, 24)

        self.intake = Intake(
            leaderCanID=DriveConstants.kIntakeLeadMotorCanId,
            followerCanID=None, leaderInverted=True, followerInverted=False,
            recoilSpeed=0.15,
            rangeFinder=self.rangefinder,
            rangeToGamepiece=100  # 100 millimeters to gamepiece at most, and if it is further away then it won't count
        )

        # The robots Elevator
        from rev import LimitSwitchConfig
        from subsystems.elevator import Elevator

        self.elevator = Elevator(
            leadMotorCANId=DriveConstants.kLeadElevationCanId,
            followMotorCANId=DriveConstants.kFollowElevationCanId,
            presetSwitchPositions=(2, 15, 28), motorClass=rev.SparkMax,
            limitSwitchType=LimitSwitchConfig.Type.kNormallyClosed,
            arm=self.arm,
            intake=self.intake
        )

        # make sure the arm respects a possibly tighter safe angle range, depending on current elevator pos
        if self.arm is not None:
            def safeArmAngleRange():
                elevatorPosition = self.elevator.getPosition()
                return constants.safeArmAngleRange(elevatorPosition)

            self.arm.setSafeAngleRangeFunction(safeArmAngleRange)

        self.elevator.setDefaultCommand(
            commands2.RunCommand(lambda: self.elevator.drive(
                self.scoringController.getRawAxis(XboxController.Axis.kRightY)
            ), self.elevator)
        )


        def maxSpeedScaledownFactor():
            if not self.elevator.zeroFound and not commands2.TimedCommandRobot.isSimulation():
                return 0.25  # if elevator does not know its zero, max speed = 25%
            elevatorPosition = self.elevator.getPosition()
            if elevatorPosition > 7.0:
                return 0.1  # if elevator position is above 7 inches, max speed = 10% (maybe needs to be much lower?)
            # otherwise, full 100%
            return 1.0

        self.robotDrive = DriveSubsystem(maxSpeedScaleFactor=maxSpeedScaledownFactor)
        if commands2.TimedCommandRobot.isSimulation():
            self.robotDrive.simPhysics = BadSimPhysics(self.robotDrive, robot)

        def onIntakeSensingGamepiece(sensing):
            if sensing:
                y = self.robotDrive.getPose().y
                if y > 4 and self.robot.isTeleop():
                    print("resetting odometry to match the left feeder location exactly")
                    self.robotDrive.resetOdometry(constants.FieldMapConstants.kLeftFeederPose)
                if y <= 4 and self.robot.isTeleop():
                    print("resetting odometry to match the right feeder location exactly")
                    self.robotDrive.resetOdometry(constants.FieldMapConstants.kRightFeederPose)

        self.intake.setOnSensingGamepiece(onIntakeSensingGamepiece)

        from subsystems.localizer import Localizer

        # tell the localizer to only allow flipped field, if it's known that the alliance color is red
        def fieldShouldBeFlipped(allianceColor: wpilib.DriverStation.Alliance):
            return allianceColor == wpilib.DriverStation.Alliance.kRed

        self.localizer = Localizer(
            drivetrain=self.robotDrive,
            fieldLayoutFile="2025-reefscape.json",
            flippedFromAllianceColor=fieldShouldBeFlipped
        )
        self.localizer.addPhotonCamera("Arducam_Front", directionDegrees=0, positionFromRobotCenter=Translation2d(x=0.30, y=0.18))
        self.localizer.addPhotonCamera("ELP_RightSide", directionDegrees=-90, positionFromRobotCenter=Translation2d(x=0.0, y=-0.30))
        self.localizer.addPhotonCamera("Arducam_Rear", directionDegrees=180, positionFromRobotCenter=Translation2d(x=-0.05, y=0.25))

        # Configure the button bindings and autos
        self.configureTrajectoryPicker()
        self.configureButtonBindings()
        self.configureAutos()

        from commands.holonomicdrive import HolonomicDrive

        # Configure drivetrain commands
        self.robotDrive.setDefaultCommand(
            # "holonomic" means that it rotates left independently of swerving left = three sticks needed to control
            HolonomicDrive(
                self.robotDrive,
                forwardSpeed=lambda: -self.driverController.getRawAxis(XboxController.Axis.kLeftY),
                leftSpeed=lambda: -self.driverController.getRawAxis(XboxController.Axis.kLeftX),
                rotationSpeed=lambda: -self.driverController.getRawAxis(XboxController.Axis.kRightX),
                deadband=OIConstants.kDriveDeadband,
                fieldRelative=True,
                rateLimit=True,
                square=True,
            )
        )


    def autonomousInit(self):
        if self.trajectoryPicker is not None:
            self.trajectoryPicker.clearDashboard()
        AutoFactory.updateDashboard(self)


    def teleopInit(self):
        AutoFactory.clearDashboard(self)
        if self.trajectoryPicker is not None:
            self.trajectoryPicker.updateDashboard()


    def configureAutos(self) -> None:
        AutoFactory.init(self)


    def configureButtonBindings(self) -> None:
        """
        Use this method to define your button->command mappings. Buttons can be created by
        instantiating a :GenericHID or one of its subclasses (CommandGenericHID or XboxController),
        and then passing it to a JoystickButton.
        """

        resetOdometryButton = self.driverController.button(XboxController.Button.kBack)
        resetOdometryButton.onTrue(ResetXY(x=0.0, y=0.0, headingDegrees=0.0, drivetrain=self.robotDrive))

        # if "start" pressed, reset X,Y position to the **lower** feeding station (x=1.30, y=6.90, 54 degrees **west**)
        startButton = self.driverController.button(XboxController.Button.kStart)
        #startButton.onTrue(ResetXY(x=1.285, y=1.135, headingDegrees=+54, drivetrain=self.robotDrive))
        startButton.onTrue(ResetXY(drivetrain=self.robotDrive, **constants.FieldMapConstants.kLeftFeeder))
        # ^^ this (x,Y) is the right feeding station for today's practice

        # if someone pushes left trigger of scoring controller more than 50%, use sticks to drive FPV
        self.configureFpvDriving(self.driverController, speed=0.3)
        if self.scoringController != self.driverController:
            self.configureFpvDriving(self.scoringController, speed=0.3)

        from commands.intakecommands import IntakeGamepiece, IntakeFeedGamepieceForward, IntakeEjectGamepieceBackward
        from commands.elevatorcommands import MoveElevatorAndArm

        # right bumper = intake new gamepiece
        intakingPosButton = self.scoringController.button(XboxController.Button.kRightBumper)
        goToIntakePositionCmd = MoveElevatorAndArm(elevator=self.elevator, position=0.0, arm=self.arm, angle=42)
        intakeCmd = IntakeGamepiece(self.intake, speed=0.115)  # .onlyIf(goToIntakePositionCmd.succeeded)
        keepWheelsLocked = RunCommand(self.robotDrive.setX, self.robotDrive)
        intakeWithWheelsLocked = goToIntakePositionCmd.andThen(intakeCmd).deadlineFor(keepWheelsLocked)

        # if rear camera is seeing feeder station, back into it first (and then run intake)
        intakeWithWheelsLocked = AutoFactory.backIntoFeeder(
            self, camera=self.rearCamera, headingDegrees=-54, speed=0.15, pushFwdSpeed=0.10, pushFwdSeconds=1.5
        ).andThen(intakeWithWheelsLocked)

        intakingPosButton.whileTrue(intakeWithWheelsLocked)

        # pull the right trigger = eject to score that gamepiece
        ejectButton = self.scoringController.axisGreaterThan(XboxController.Axis.kRightTrigger, 0.5)
        ejectForwardCmd = IntakeFeedGamepieceForward(self.intake, speed=0.3).withTimeout(0.3)
        ejectButton.whileTrue(ejectForwardCmd)

        # pull the left trigger = spin the intake in reverse direction
        ejectBackwardsButton = self.scoringController.axisGreaterThan(XboxController.Axis.kLeftTrigger, 0.5)
        ejectBackwards = IntakeEjectGamepieceBackward(self.intake, speed=0.3).withTimeout(0.3)
        ejectBackwardsButton.whileTrue(ejectBackwards)

        # elevator buttons for different levels
        #  - 0
        level0PosButton = self.scoringController.button(XboxController.Button.kA)
        level0PositionCmd = MoveElevatorAndArm(elevator=self.elevator, position=0.0, arm=self.arm, angle=70)
        level0PosButton.onTrue(level0PositionCmd)
        # (in game manual there are levels 2, 3 and 4)
        #  - 2
        level2PosButton = self.scoringController.button(XboxController.Button.kB)
        level2PositionCmd = MoveElevatorAndArm(elevator=self.elevator, position= 4.0, arm=self.arm, angle=ArmConstants.kArmSafeStartingAngle)
        level2PosButton.onTrue(level2PositionCmd)
        #  - 3
        level3PosButton = self.scoringController.button(XboxController.Button.kY)
        level3PositionCmd = MoveElevatorAndArm(elevator=self.elevator, position= 13.0, arm=self.arm, angle=ArmConstants.kArmSafeStartingAngle)
        level3PosButton.onTrue(level3PositionCmd)
        #  - 4
        level4PosButton = self.scoringController.button(XboxController.Button.kX)
        level4PositionCmd = MoveElevatorAndArm(elevator=self.elevator, position= 30.0, arm=self.arm, angle=135)
        level4PosButton.onTrue(level4PositionCmd)

        # right and left POV of scoring joystick = aligning to AprilTags (using current, imprecise, robot heading!)
        if self.scoringController != self.driverController:
            self.scoringController.povLeft().whileTrue(self.alignToTagCmd(self.frontRightCamera, None))
            self.scoringController.povRight().whileTrue(self.alignToTagCmd(self.frontLeftCamera, None))


    def configureFpvDriving(self, joystick, speed):
        """
        FPV = not field-relative
        """
        from commands.holonomicdrive import HolonomicDrive

        # if someone pushes this left bumper
        leftBumper = joystick.button(XboxController.Button.kLeftBumper)

        # ... then the sticks of this joystick start driving the robot FPV-style (not field-relative)
        leftBumper.whileTrue(
            HolonomicDrive(
                self.robotDrive,
                forwardSpeed=lambda: -speed * joystick.getRawAxis(XboxController.Axis.kLeftY),
                leftSpeed=lambda: -speed * joystick.getRawAxis(XboxController.Axis.kLeftX),
                rotationSpeed=lambda: -speed * joystick.getRawAxis(XboxController.Axis.kRightX),
                deadband=0,
                fieldRelative=False,  # driving FPV (first person view), not field-relative (install an FPV camera on robot?)
                rateLimit=False,
                square=True,
            )
        )


    def configureTrajectoryPicker(self, swerve="last-point"):
        from commands.trajectory_picker import TrajectoryPicker, ReversedTrajectoryPicker

        # trajectory picker will only run when these subsystems are not busy with other commands
        requirements = [self.robotDrive, self.intake, self.arm, self.elevator]

        # POV up: run the trajectory while button pushed
        self.trajectoryPicker = TrajectoryPicker(self.robotDrive.field, subsystems=requirements)
        self.driverController.povUp().whileTrue(self.trajectoryPicker)

        # POV left+right: pick trajectory
        self.driverController.povLeft().onTrue(InstantCommand(self.trajectoryPicker.previousTrajectory))
        self.driverController.povRight().onTrue(InstantCommand(self.trajectoryPicker.nextTrajectory))

        backUp = SwerveToSide(metersToTheLeft=0, metersBackwards=0.3, drivetrain=self.robotDrive, speed=0.2)
        armDown = MoveElevatorAndArm(self.elevator, position=0.0, arm=self.arm, angle=42)

        # POV down: run the reverse trajectory while pushed
        self.reversedTrajectoryPicker = ReversedTrajectoryPicker(self.trajectoryPicker, subsystems=[self.robotDrive])
        # (may as well bring that arm down along with driving in reverse)
        reverseTrajectoryWithArmGoingDown = self.reversedTrajectoryPicker.alongWith(armDown)
        # (when button is pushed, first back up safely and then drive the reverse trajectory)
        self.driverController.povDown().whileTrue(backUp.andThen(reverseTrajectoryWithArmGoingDown))

        # a function to choose trajectory by combining the letter and side (for example, "C-left")
        def chooseTrajectory(letter=None, side=None):
            if letter:
                print(f"choosing trajectory letter {letter}")
                self.trajectoryLetter = letter
            if side:
                print(f"choosing trajectory side {side}")
                self.trajectorySide = side
            self.trajectoryPicker.pickTrajectory(self.trajectoryLetter + "-" + self.trajectorySide)

        # trajectory board using this function (are the button numbers correct?)
        self.trajectoryBoard.button(5).onTrue(InstantCommand(lambda: chooseTrajectory(letter="A")))
        self.trajectoryBoard.button(6).onTrue(InstantCommand(lambda: chooseTrajectory(letter="B")))
        self.trajectoryBoard.button(7).onTrue(InstantCommand(lambda: chooseTrajectory(letter="C")))
        self.trajectoryBoard.button(8).onTrue(InstantCommand(lambda: chooseTrajectory(letter="D")))
        self.trajectoryBoard.button(9).onTrue(InstantCommand(lambda: chooseTrajectory(letter="E")))
        self.trajectoryBoard.button(10).onTrue(InstantCommand(lambda: chooseTrajectory(letter="F")))
        self.trajectoryBoard.button(11).onTrue(InstantCommand(lambda: chooseTrajectory(side="left")))
        self.trajectoryBoard.button(12).onTrue(InstantCommand(lambda: chooseTrajectory(side="right")))

        # while the "left" and "right" buttons of trajectory board are pressed, keep driving the chosen trajectory
        self.trajectoryBoard.button(11).whileTrue(self.trajectoryPicker)
        self.trajectoryBoard.button(12).whileTrue(self.trajectoryPicker)

        # when moving on a reversed trajectory, robot can prepare to back into left feeder or right feeder
        def prepareToBackIntoLeftFeeder():
            self.rearCamera.setOnlyTagIds([1, 13])

        def prepareToBackIntoRightFeeder():
            self.rearCamera.setOnlyTagIds([2, 12])


        # now add the trajectories (please replace these with the real ones):

        # feeder locations:
        #startButton.onTrue(ResetXY(x=1.285, y=1.135, headingDegrees=+54, drivetrain=self.robotDrive))
        #startButton.onTrue(ResetXY(x=1.285, y=6.915, headingDegrees=-54, drivetrain=self.robotDrive))

        #  - go to left branch of reef side B
        goSideELeftBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(5.594, 5.635, -120),
            waypoints=[
                (1.285, 6.915, -54.0),
                (2.641, 5.922, -40),
                (4.806, 6.943, -90),
            ],
            speed=0.2,
            setup=prepareToBackIntoLeftFeeder,
        )
        self.trajectoryPicker.addCommands(
            "E-left",
            goSideELeftBranch,
            self.alignToTagCmd(self.frontRightCamera, desiredHeading=+240)
        )

        goSideERightBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(5.165, 5.606, -120),
            waypoints=[
                (1.285, 6.915, -54.0),
                (2.641, 5.922, -40),
                (4.306, 6.643, -75),
            ],
            speed=0.4,
            setup=prepareToBackIntoLeftFeeder,
        )
        self.trajectoryPicker.addCommands(
            "E-right",
            goSideERightBranch,
            self.alignToTagCmd(self.frontLeftCamera, desiredHeading=+240)
        )

        #  - go to right branch of reef side B
        goSideCLeftBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(5.045, 2.611, 120.0),
            waypoints=[
                (1.285, 1.135, 54.0),
                (2.336, 1.911, -10.119),
                (3.777, 1.520, 0.302),
                (5.045, 1.741, 50.001),
            ],
            speed=0.2,
            setup=prepareToBackIntoRightFeeder,
        )
        self.trajectoryPicker.addCommands(
            "C-left",
            goSideCLeftBranch,
            self.alignToTagCmd(self.frontRightCamera, desiredHeading=+120)
        )

        goSideCRightBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(5.454, 2.724, 120.0),
            waypoints=[
                (1.285, 1.135, 54.0),
                (2.336, 1.911, -10.119),
                (3.777, 1.520, 0.302),
                (5.045, 1.741, 50.001),
            ],
            speed=0.2,
            setup=prepareToBackIntoRightFeeder,
        )
        self.trajectoryPicker.addCommands(
            "C-right",
            goSideCRightBranch,
            self.alignToTagCmd(self.frontLeftCamera, desiredHeading=+120)
        )

        goSideALeftBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(2.47, 4.250, 0),
            waypoints=[
                (1.285, 6.915, -54),
                (1.641, 5.922, -54),
            ],
            speed=0.1,
            setup=prepareToBackIntoLeftFeeder,
        )
        self.trajectoryPicker.addCommands(
            "A-left",
            goSideALeftBranch,
           self.alignToTagCmd(self.frontRightCamera, desiredHeading=0)
        )
        goSideARightBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(2.472, 3.501, 0),
            waypoints=[
                (1.285, 6.915, -54),
                (1.641, 5.922, -54),
            ],
            speed=0.1,
            setup=prepareToBackIntoLeftFeeder,
        )
        self.trajectoryPicker.addCommands(
            "A-right",
            goSideARightBranch,
            self.alignToTagCmd(self.frontLeftCamera, desiredHeading=0)
        )

        goSideBLeftBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(3.450, 2.574, 60.0),
            waypoints=[
                (1.285, 1.135, 54.0),
                (2.201, 1.986, 54.0),
            ],
            speed=0.2,
            setup=prepareToBackIntoRightFeeder,
        )
        self.trajectoryPicker.addCommands(
            "B-left",
            goSideBLeftBranch,
            self.alignToTagCmd(self.frontRightCamera, desiredHeading=+60)
        )



        goSideBRightBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(3.660, 2.265, 60.0),
            waypoints=[
                (1.285, 1.135, 54.0),
                (2.201, 1.986, 54.0),
            ],
            speed=0.2,
            setup=prepareToBackIntoRightFeeder,
        )
        self.trajectoryPicker.addCommands(
            "B-right",
            goSideBRightBranch,
            self.alignToTagCmd(self.frontLeftCamera, desiredHeading=+60)
        )

        goSideDLeftBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(6.202, 3.791, 180),
            waypoints=[
                (1.285, 1.135, 54.0),
                (2.201, 1.986, 54.0),
                (5.155, 1.516, 90),
                (6.402, 2.694, 135),
            ],
            speed=0.2,
            setup=prepareToBackIntoRightFeeder,
        )
        self.trajectoryPicker.addCommands(
            "D-left",
            goSideDLeftBranch,
            self.alignToTagCmd(self.frontRightCamera, desiredHeading=+180)
        )

        goSideDRightBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(6.252, 4.180, 180),
            waypoints=[
                (1.285, 1.135, 54.0),
                (2.201, 1.986, 54.0),
                (4.477, 1.306, 90),
                (6.482, 2.824, 135),
            ],
            speed=0.2,
            setup=prepareToBackIntoRightFeeder,
        )
        self.trajectoryPicker.addCommands(
            "D-right",
            goSideDRightBranch,
            self.alignToTagCmd(self.frontLeftCamera, desiredHeading=+180)
        )

        goSideFLeftBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(3.600, 5.546, -60.0),
            waypoints=[
                (1.285, 6.915, -54),
                (2.641, 5.922, -40),
            ],
            speed=0.2,
            setup=prepareToBackIntoLeftFeeder,
        )

        self.trajectoryPicker.addCommands(
            "F-left",
            goSideFLeftBranch,
            self.alignToTagCmd(self.frontRightCamera, desiredHeading=+300)
        )
        goSideFRightBranch = JerkyTrajectory(
            drivetrain=self.robotDrive,
            swerve=swerve,
            endpoint=(3.470, 5.446, -60.0),
            waypoints=[
                (1.285, 6.915, -54),
                (2.641, 5.922, -40),
            ],
            speed=0.2,
            setup=prepareToBackIntoLeftFeeder,
        )
        self.trajectoryPicker.addCommands(
            "F-right",
            goSideFRightBranch,
            self.alignToTagCmd(self.frontLeftCamera, desiredHeading=+300)
        )

    def disablePIDSubsystems(self) -> None:
        """Disables all ProfiledPIDSubsystem and PIDSubsystem instances.
        This should be called on robot disable to prevent integral windup."""

    def getAutonomousCommand(self) -> commands2.Command:
        """
        :returns: the command to run in autonomous
        """
        return AutoFactory.makeAutoCommand(self)


    def getTestCommand(self) -> typing.Optional[commands2.Command]:
        """
        :returns: the command to run in test mode (to exercise all systems)
        """
        from commands.intakecommands import IntakeGamepiece, IntakeFeedGamepieceForward
        from commands.elevatorcommands import MoveElevatorAndArm
        from commands.aimtodirection import AimToDirection

        # 1. intake the gamepiece and eject it in position 2, to test arm+elevator+intake
        intake = MoveElevatorAndArm(position=0, angle=42, elevator=self.elevator, arm=self.arm).andThen(
            IntakeGamepiece(intake=self.intake, speed=0.115).withTimeout(10.0)
        )
        score = MoveElevatorAndArm(position=13.0, elevator=self.elevator, arm=self.arm).andThen(
            IntakeFeedGamepieceForward(intake=self.intake, speed=0.3).withTimeout(0.3)
        )
        drop = MoveElevatorAndArm(position=0, angle=42, elevator=self.elevator, arm=self.arm)

        # 2. some rotations to test the gyro
        rotation1 = AimToDirection(degrees=60, speed=0.3, drivetrain=self.robotDrive)
        rotation2 = AimToDirection(degrees=0.0, speed=0.3, drivetrain=self.robotDrive)
        rotations = rotation1.andThen(rotation2)

        # 3. square dance to test the drivetrain
        from commands.swervetopoint import SwerveToSide
        forward = SwerveToSide(metersToTheLeft=0, metersBackwards=-0.5, speed=0.2, drivetrain=self.robotDrive)
        left = SwerveToSide(metersToTheLeft=0.5, metersBackwards=0, speed=0.2, drivetrain=self.robotDrive)
        back = SwerveToSide(metersToTheLeft=0, metersBackwards=0.5, speed=0.2, drivetrain=self.robotDrive)
        right = SwerveToSide(metersToTheLeft=-0.5, metersBackwards=0, speed=0.2, drivetrain=self.robotDrive)
        squareDance = forward.andThen(left).andThen(back).andThen(right)

        # 4. not yet done: add commands for tag alignment with both cameras?

        # 5. the combination
        return intake.andThen(score).andThen(drop).andThen(rotations).andThen(squareDance)


    def alignToTagCmd(self, camera, desiredHeading):
        from commands.setcamerapipeline import SetCameraPipeline
        from commands.followobject import FollowObject, StopWhen
        from commands.alignwithtag import AlignWithTag
        from commands.swervetopoint import SwerveToSide

        # switch to camera pipeline 3, to start looking for certain kind of AprilTags
        lookForTheseTags = SetCameraPipeline(camera, 1)
        approachTheTag = FollowObject(camera, self.robotDrive, stopWhen=StopWhen(maxSize=10), speed=0.1)  # stop when tag size=4 (4% of the frame pixels)

        # TODO: add timeout
        alignAndPush = AlignWithTag(camera, self.robotDrive, desiredHeading, speed=0.2, pushForwardSeconds=1.5, pushForwardSpeed=0.07).withTimeout(8)

        # connect them together
        alignToScore = lookForTheseTags.andThen(approachTheTag).andThen(alignAndPush)

        # or you can do this, if you want to score the coral 15 centimeters to the right and two centimeters back from the AprilTag
        # stepToSide = SwerveToSide(drivetrain=self.robotDrive, metersToTheLeft=-0.15, metersBackwards=0.02, speed=0.2)
        # alignToScore = lookForTheseTags.andThen(approachTheTag).andThen(alignAndPush).andThen(stepToSide)

        return alignToScore
