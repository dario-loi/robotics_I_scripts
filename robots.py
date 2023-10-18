#!/usr/bin/env python3

import typer
from enum import Enum
from typing import Optional, Union, Tuple, Type, List, Annotated, Any, Callable
import pyrobots as pr
from rich import print
from sympy import Matrix, sympify, nsimplify
from fractions import Fraction
import numpy as np

app = typer.Typer(
    help="Robotics toolbox for Python.\n\nMathematical expressions such as sqrt(2) or pi are supported and will be evaluated whenever a number is to be inserted",
    no_args_is_help=True,
)


class RotProblemType(Enum):
    DIRECT = "direct"
    INVERSE = "inverse"


Expression = Annotated[str, typer.Argument(..., help="Numerical expression.")]
Axis = Annotated[str, typer.Argument(..., help="Rotation axis.")]
NDarray = Annotated[str, typer.Argument(..., help="Numerical ndarray.")]
Float = Union[float, np.float_]


def eval_expr(expr: Expression) -> float:
    try:
        val: float = float(sympify(expr))
        return val
    except:
        raise typer.BadParameter("Provided input is not a valid expression.")


def parse_axis(axis_str: Axis) -> np.ndarray:
    match axis_str:
        case "x":
            return np.array([1, 0, 0])
        case "y":
            return np.array([0, 1, 0])
        case "z":
            return np.array([0, 0, 1])
        case _:
            try:
                values = Matrix(sympify(axis_str))
                return np.array(values).astype(np.float64)
            except ValueError:
                raise typer.BadParameter("Provided input is not a valid axis.")


def parse_ndarray(input_str: NDarray) -> np.ndarray:
    try:
        values = Matrix(sympify(input_str))
        return np.array(values).astype(np.float64)
    except ValueError:
        raise typer.BadParameter("Provided input is not a valid ndarray.")


def format_known_or(x: Float) -> str:
    # Some masterful code right here

    prefix = ""
    if x < 0:
        prefix = "-"

    if np.isclose(np.abs(x), np.pi):
        return prefix + "π"
    elif np.isclose(np.abs(x), np.pi / 2):
        return prefix + "π/2"
    elif np.isclose(np.abs(x), np.pi / 4):
        return prefix + "π/4"
    elif np.isclose(np.abs(x), np.pi / 3):
        return prefix + "π/3"
    elif np.isclose(np.abs(x), np.sqrt(2)):
        return prefix + "√2"
    elif np.isclose(np.abs(x), 1 / np.sqrt(2)):
        return prefix + "1/√2"
    elif np.isclose(np.abs(x), np.sqrt(3)):
        return prefix + "√3"
    else:
        return str(Fraction(float(x)).limit_denominator())


@app.command()
def rotation_interactive(
    fract: Annotated[
        bool,
        typer.Option(
            ...,
            "--fract",
            "-f",
            help="Use fractions instead of decimals.",
            show_default=False,
        ),
    ] = False
) -> None:
    """
    Interactive interface to solve rotation problems.
    """

    problem_str: str = typer.prompt(
        "What class of problem do you want to solve? [direct,inverse]", type=str
    )
    problem_type = RotProblemType(problem_str.strip().lower())

    match problem_type:
        case RotProblemType.DIRECT:
            theta_str: Expression = typer.prompt(
                "Rotation angle (in radians):", type=Expression, default=0
            )
            axis_str: Expression = typer.prompt(
                "Rotation axis (Either a list or x,y or z for a unit axis):",
                default="1,0,0",
            )
            theta: float = eval_expr(theta_str)
            axis: np.ndarray = parse_axis(axis_str)

            R = pr.rotations.direct_rot_mat(theta, axis)

            if fract:
                R = np.array2string(
                    R,
                    separator=", ",
                    suppress_small=True,
                    formatter={"float": lambda x: format_known_or(x)},
                )
            else:
                R = np.array2string(R, separator=", ")

            print(f"Direct Rotation Matrix:\n{R}")

        case RotProblemType.INVERSE:
            R_str: Expression = typer.prompt(
                "Rotation matrix (in the form [[a, b, c], [d, e, f], [g, h, i]]):",
                default="[[1,0,0],[0,1,0],[0,0,1]]",
            )
            R_out: np.ndarray = parse_ndarray(R_str)
            theta, axis = pr.rotations.inverse_rot_mat(R_out)
            axis_fmt: str = (
                np.array2string(axis, separator=", ")
                if isinstance(axis, np.ndarray)
                else "Undefined"
            )
            print(f"Inverse Rotation:\ntheta = {theta}\naxis = {axis_fmt}")


@app.command()
def rotation_direct(
    theta: Expression,
    axis: Axis,
    fract: Annotated[
        bool,
        typer.Option(
            ...,
            "--fract",
            "-f",
            help="Use fractions instead of decimals.",
            show_default=False,
        ),
    ] = False,
) -> None:
    """
    Direct rotation problem.
    """

    parsed_axis: np.ndarray = parse_axis(axis)
    parsed_theta: float = eval_expr(theta)
    R = pr.rotations.direct_rot_mat(parsed_theta, parsed_axis)
    if fract:
        R = np.array2string(
            R,
            separator=", ",
            suppress_small=True,
            formatter={"float": lambda x: format_known_or(x)},
        )
    else:
        R = np.array2string(R, separator=", ")
    print(f"Direct Rotation Matrix:")
    print(R)


@app.command()
def rotation_inverse(r_matrix: NDarray):
    """
    Inverse rotation problem.
    """
    parsed_R: np.ndarray = parse_ndarray(r_matrix)
    theta, axis = pr.rotations.inverse_rot_mat(parsed_R)
    axis = (
        np.array2string(axis, separator=", ")
        if isinstance(axis, np.ndarray)
        else "Undefined"
    )

    print(f"Inverse Rotation:\ntheta = {theta}\naxis = {axis}")
    
@app.command()
def rpy_direct(
    roll: Expression,
    pitch: Expression,
    yaw: Expression,
    fract: Annotated[
        bool,
        typer.Option(
            ...,
            "--fract",
            "-f",
            help="Use fractions instead of decimals.",
            show_default=False,
        ),
    ] = False,
    separate: Annotated[
        bool,
        typer.Option(
            ...,
            "--separate",
            "-s",
            help="Separate the rotation matrix into roll, pitch and yaw.",
            show_default=False,
        ),
    ] = False,
) -> None:
    """
    Direct rotation problem.
    """

    parsed_roll: float = eval_expr(roll)
    parsed_pitch: float = eval_expr(pitch)
    parsed_yaw: float = eval_expr(yaw)
    
    if separate:
        R_roll, R_pitch, R_yaw = pr.rotations.direct_rpy_separate(parsed_roll, parsed_pitch, parsed_yaw)
        if fract:
            R_roll = np.array2string(
                R_roll,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
            R_pitch = np.array2string(
                R_pitch,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
            R_yaw = np.array2string(
                R_yaw,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
        else:
            R_roll = np.array2string(R_roll, separator=", ")
            R_pitch = np.array2string(R_pitch, separator=", ")
            R_yaw = np.array2string(R_yaw, separator=", ")
        print(f"Direct Rotation Matrix:")
        print(f"Roll:\n{R_roll}")
        print(f"Pitch:\n{R_pitch}")
        print(f"Yaw:\n{R_yaw}")
    else:
        R = pr.rotations.direct_rpy(parsed_roll, parsed_pitch, parsed_yaw)
        if fract:
            R = np.array2string(
                R,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
        else:
            R = np.array2string(R, separator=", ")
        print(f"Direct Rotation Matrix:")
        print(R)
    



if __name__ == "__main__":
    app()
