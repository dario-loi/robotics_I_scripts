#!/usr/bin/env python3

import typer
from enum import Enum
from typing import Optional, Union, Tuple, Type, List, Annotated, Any
import pyrobots as pr
from rich import print
from sympy import Matrix, sympify
import numpy as np

app = typer.Typer(help="Robotics toolbox for Python.\n\nMathematical expressions such as sqrt(2) or pi are supported and will be evaluated whenever a number is to be inserted", no_args_is_help=True)

class RotProblemType(Enum):
    DIRECT = "direct"
    INVERSE = "inverse"
    

Expression = Annotated[str, typer.Argument(..., help="Numerical expression.")]
Axis = Annotated[str, typer.Argument(..., help="Rotation axis.")]
NDarray = Annotated[str, typer.Argument(..., help="Numerical ndarray.")]

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
    
def mat_3d_commas(ndarr_str : np.ndarray) -> str:
    # Reimplement formatting to get commas in the right place
    
    arr_s = repr(ndarr_str)
    arr_s = arr_s.lstrip("array(").rstrip(")")
    arr_s = [line.strip() for line in arr_s.split("\n")]
    arr_s[1] = " " + arr_s[1]
    arr_s[2] = " " + arr_s[2]
    arr_s = "\n".join(arr_s)
    return arr_s

@app.command()
def rotation_interactive():
    """
        Interactive interface to solve rotation problems.
    """
    
    problem_type: str = typer.prompt("What class of problem do you want to solve? [direct,inverse]", type=str)
    problem_type = RotProblemType(problem_type.strip().lower())
    
    match problem_type:
        case RotProblemType.DIRECT:
            theta: Expression = typer.prompt("Rotation angle (in radians):", type=Expression, default=0)
            axis_str: Expression = typer.prompt("Rotation axis (Either a list or x,y or z for a unit axis):", default="1,0,0")
            theta = eval_expr(theta)
            axis: np.ndarray = parse_axis(axis_str)
            print(f"Direct Rotation Matrix:\n{mat_3d_commas(pr.rotations.direct_rot_mat(theta, axis))}")

        case RotProblemType.INVERSE:
            R_str: Expression = typer.prompt("Rotation matrix (in the form [[a, b, c], [d, e, f], [g, h, i]]):", default="[[1,0,0],[0,1,0],[0,0,1]]")
            R: np.ndarray = parse_ndarray(R_str)
            theta, axis = pr.rotations.inverse_rot_mat(R)
            print(f"Inverse Rotation:\ntheta = {theta}\naxis = {axis}")
            
@app.command()
def rotation_direct(theta: Expression, axis: Axis):
    """
        Direct rotation problem.
    """
    
    parsed_axis: np.ndarray = parse_axis(axis)
    parsed_theta: float = eval_expr(theta)
    R = mat_3d_commas(pr.rotations.direct_rot_mat(parsed_theta, parsed_axis))
    print(f"Direct Rotation Matrix:\n{R}")

    
@app.command()
def rotation_inverse(r_matrix: NDarray):
    """
        Inverse rotation problem.
    """
    parsed_R: np.ndarray = parse_ndarray(r_matrix)
    theta, axis = pr.rotations.inverse_rot_mat(parsed_R)
    if not isinstance(axis, np.ndarray):
        axis = "Undefined"
    
    print(f"Inverse Rotation:\ntheta = {theta}\naxis = {axis}")

if __name__ == "__main__":
    app()