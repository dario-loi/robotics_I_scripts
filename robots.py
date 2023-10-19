#!/usr/bin/env python3

from enum import Enum
from fractions import Fraction
from typing import Annotated, Any, Callable, List, Optional, Tuple, Type, Union

import numpy as np
from numpy import floating
import typer
from rich import print
from rich.prompt import Prompt
from sympy import Matrix, nsimplify, sympify

import pyrobots as pr

app = typer.Typer(
    help="Robotics toolbox for Python.\n\nMathematical expressions such as sqrt(2) or pi are supported and will be evaluated whenever a number is to be inserted\n\nIf you need to solve for complex problems, import the library in a notebook and DYI!",
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


def format_known_or(x: floating[Any]) -> str:
    def is_close_to(value, reference, tolerance=1e-10):
        return np.isclose(np.abs(value), reference, atol=tolerance)

    def format_special_case(value, symbol):
        return f"{prefix}{symbol}"

    prefix = "-" if x < 0 else ""

    special_cases = {
        np.pi: "π",
        np.pi / 2: "π/2",
        np.pi / 4: "π/4",
        np.pi / 3: "π/3",
        np.sqrt(2): "√2",
        1 / np.sqrt(2): "1/√2",
        np.sqrt(3): "√3",
        1 / np.sqrt(3): "1/√3",
        np.e: "e",
        2 * np.pi: "2π",
        np.pi / 6: "π/6",
        np.sqrt(5): "√5",
        1 / np.sqrt(5): "1/√5",
        np.log(10): "ln(10)",
        np.log(2): "ln(2)",
    }

    for reference, symbol in special_cases.items():
        if is_close_to(x, reference):
            return format_special_case(x, symbol)
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

    problem_str: str = Prompt.ask(
        ":thinking_face: What class of problem do you want to solve? (direct,inverse)",
    )
    problem_type = RotProblemType(problem_str.strip().lower())
    R_str: str

    match problem_type:
        case RotProblemType.DIRECT:
            theta_str: Expression = Prompt.ask(
                ":triangular_ruler: Rotation angle (in radians)", default="0"
            )
            axis_str: Expression = Prompt.ask(
                ":straight_ruler: Rotation axis (Either a list or x,y or z for a unit axis)",
                default="[1,0,0]",
            )
            theta = eval_expr(theta_str)
            axis = parse_axis(axis_str)

            R = pr.rotations.direct_rot_mat(theta, axis)

            if fract:
                R_str = np.array2string(
                    R,
                    separator=", ",
                    suppress_small=True,
                    formatter={"float": lambda x: format_known_or(x)},
                )
            else:
                R_str = np.array2string(R, separator=", ")

            print(f":arrows_counterclockwise: Direct Rotation Matrix:\n{R_str}")

        case RotProblemType.INVERSE:
            R_in: Expression = Prompt.ask(
                ":arrows_counterclockwise: Rotation matrix (in the form [[a, b, c], [d, e, f], [g, h, i]]):",
                default="[[1,0,0],[0,1,0],[0,0,1]]",
            )
            R_out = parse_ndarray(R_in)
            theta_out, axis_out = pr.rotations.inverse_rot_mat(R_out)
            axis_fmt = (
                np.array2string(axis_out, separator=", ")
                if isinstance(axis_out, np.ndarray)
                else "Undefined"
            )
            print(
                f"Inverse Rotation:\n:triangular_ruler: theta = {theta_out}\n:straight_ruler: axis = {axis_fmt}"
            )


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
    R_str: str
    if fract:
        R_str = np.array2string(
            R,
            separator=", ",
            suppress_small=True,
            formatter={"float": lambda x: format_known_or(x)},
        )
    else:
        R_str = np.array2string(R, separator=", ")
    print(f":arrows_counterclockwise: Direct Rotation Matrix:\n{R_str}")


@app.command()
def rotation_inverse(r_matrix: NDarray):
    """
    Inverse rotation problem.
    """
    parsed_R: np.ndarray = parse_ndarray(r_matrix)
    theta, axis = pr.rotations.inverse_rot_mat(parsed_R)
    axis_fmt = (
        np.array2string(axis, separator=", ")
        if isinstance(axis, np.ndarray)
        else "Undefined"
    )

    print(
        f"Inverse Rotation:\n:triangular_ruler: theta = {theta}\n:straight_ruler: axis = {axis_fmt}"
    )


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
        R_roll, R_pitch, R_yaw = pr.rotations.direct_rpy_separate(
            parsed_roll, parsed_pitch, parsed_yaw
        )
        
        roll_fmt: str
        pitch_fmt: str
        yaw_fmt: str
        
        if fract:
            roll_fmt = np.array2string(
                R_roll,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
            pitch_fmt = np.array2string(
                R_pitch,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
            yaw_fmt = np.array2string(
                R_yaw,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
        else:
            roll_fmt = np.array2string(R_roll, separator=", ")
            pitch_fmt = np.array2string(R_pitch, separator=", ")
            yaw_fmt = np.array2string(R_yaw, separator=", ")
            
        print(f"Direct Rotation Matrix:")
        print(f":arrow_lower_right: Roll:\n{roll_fmt}")
        print(f":arrow_lower_left: Pitch:\n{pitch_fmt}")
        print(f":arrow_up: Yaw:\n{yaw_fmt}")
    else:
        R = pr.rotations.direct_rpy(parsed_roll, parsed_pitch, parsed_yaw)
        R_fmt: str
        if fract:
            R_fmt = np.array2string(
                R,
                separator=", ",
                suppress_small=True,
                formatter={"float": lambda x: format_known_or(x)},
            )
        else:
            R_fmt = np.array2string(R, separator=", ")
        print(f":arrows_counterclockwise: Direct Rotation Matrix:\n{R_fmt}")


if __name__ == "__main__":
    app()
