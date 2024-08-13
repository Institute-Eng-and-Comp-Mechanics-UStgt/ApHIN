import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import math
from PIL import Image
import itertools
from copy import copy


# %% create circular shape
def create_pixel_circle(center_point, pixel_array, r=2):

    debug_mode = False
    round_value = 8

    if debug_mode:
        # # define circle
        # r = 20
        # # center in origin
        # center_point = np.array([r+1.95,r+1.55])
        # num_pixels = np.array([np.ceil(center_point[0]).astype(int)+r+1,np.ceil(center_point[1]).astype(int)+r+1])
        r = r
        center_point = center_point

    # closed form integral
    def circle_area_segment(r, a, b, axis=0):
        """
        calculates area of a circle segment in closed form which is defined through the x or y-boundaries [a,b]
        :param r: radius of the circle
        :param a: lower integration boundary
        :param b: upper integration boundary
        :param segment: (int) '0' or '1' integration over x=0 or y=1
        :return: area of the segment
        """
        assert a <= r and a >= -r
        assert b <= r and b >= -r
        assert b >= a

        if axis == 0:
            primitive_function = lambda x: -1 / 2 * (r**2) * np.arccos(
                x / r
            ) + 1 / 2 * x * math.sqrt(r**2 - x**2)
        elif axis == 1:
            primitive_function = lambda y: 1 / 2 * (r**2) * np.arcsin(
                y / r
            ) + 1 / 2 * y * math.sqrt(r**2 - y**2)

        # insert boundaries
        area = primitive_function(b) - primitive_function(a)
        return area

    def transform_to_relative_coords(coords, center_point, axis=None, round_value=8):
        """
        relative coordinates where circle center lies in the origin
        :param coords: (array or scalar) coordinates to transform
        :param center_point: center point of circle
        :param axis: transform either [x,y] or only 'x'(0) or only 'y'(1)
        """
        if axis is None:
            rel_coords = coords - center_point
        elif axis == 0:
            # only x-coordinate
            rel_coords = coords - center_point[0]
        elif axis == 1:
            # only y-coordinate
            rel_coords = coords - center_point[1]
            # round because of floating point issues - leading to higher values than radius
        return np.around(rel_coords, round_value)

    def transform_to_absolute_coords(coords, center_point, axis=None, round_value=8):
        """
        absolute coordinates from coordinates where circle center lies in the origin
        :param coords: (array or scalar) coordinates to transform
        :param center_point: center point of circle
        :param axis: transform either [x,y] or only 'x'(0) or only 'y'(1)
        """
        if axis is None:
            abs_coords = coords + center_point
        elif axis == 0:
            # only x-coordinate
            abs_coords = coords + center_point[0]
        elif axis == 1:
            # only y-coordinate
            abs_coords = coords + center_point[1]
            # round because of floating point issues - leading to higher values than radius
        return np.around(abs_coords, round_value)

    def get_counter_value(grid_value, r):
        """
        calculates x if y is inserted and vice versa
        :param r: radius of the circle
        :param grid value: either x or y value
        :return: counter value, x when y is inserted and vice versa
        """
        counter_value_pos = math.sqrt(r**2 - grid_value**2)
        counter_value_neg = -math.sqrt(r**2 - grid_value**2)

        return counter_value_pos, counter_value_neg

    def fraction_to_color(fraction, only_white_part=False):
        """
        calculates the grey-scale value from the area fraction
        :param fraction: area fraction of pixel [0,1]
        :param only_white_part: get only the white part, which is subtracted
        :return: color value [0,255]
        """
        if np.abs(fraction) < 1e-6:
            fraction = 0
        assert fraction >= 0 and fraction <= 1
        color_white = 255
        color_black = 0
        if only_white_part:
            return fraction * color_white
        else:
            return color_white - fraction * color_white

    # %% get all intersection points between circle and grid
    # loop first over x and afterwards over y grid values and get counter value
    # looping over x
    left_boundary = center_point[0] - r
    right_boundary = center_point[0] + r
    # initialize grid point
    grid_point_x = left_boundary
    # initialize intersection coordinates
    intersection_coords = []
    while grid_point_x >= left_boundary and grid_point_x <= right_boundary:
        # from left to right
        grid_point_x = np.ceil(grid_point_x)
        grid_point_x_rel = transform_to_relative_coords(
            grid_point_x, center_point, axis=0
        )
        y_value_grid_x_pos_rel, y_value_grid_x_neg_rel = get_counter_value(
            grid_point_x_rel, r
        )
        y_value_grid_x_pos_abs = transform_to_absolute_coords(
            y_value_grid_x_pos_rel, center_point, axis=1
        )
        y_value_grid_x_neg_abs = transform_to_absolute_coords(
            y_value_grid_x_neg_rel, center_point, axis=1
        )
        if y_value_grid_x_pos_rel != y_value_grid_x_neg_rel:
            # use only one intersection point
            intersection_coords.append((grid_point_x, y_value_grid_x_neg_abs))
        intersection_coords.append((grid_point_x, y_value_grid_x_pos_abs))
        # increase grid point value
        grid_point_x += 1

    # looping over y
    top_boundary = center_point[1] + r
    bottom_boundary = center_point[1] - r
    # initialize grid point
    grid_point_y = bottom_boundary
    while grid_point_y >= bottom_boundary and grid_point_y <= top_boundary:
        # from left to right
        grid_point_y = np.ceil(grid_point_y)
        grid_point_y_rel = transform_to_relative_coords(
            grid_point_y, center_point, axis=1
        )
        x_value_grid_y_pos, x_value_grid_y_neg = get_counter_value(grid_point_y_rel, r)
        pos_tuple_y = (
            transform_to_absolute_coords(x_value_grid_y_pos, center_point, axis=0),
            grid_point_y,
        )
        neg_tuple_y = (
            transform_to_absolute_coords(x_value_grid_y_neg, center_point, axis=0),
            grid_point_y,
        )

        if x_value_grid_y_pos != x_value_grid_y_neg:
            # use only one intersection point
            if pos_tuple_y not in intersection_coords:
                intersection_coords.append(pos_tuple_y)
        if neg_tuple_y not in intersection_coords:
            intersection_coords.append(neg_tuple_y)
        grid_point_y += 1

    intersection_coords_array = np.around(np.array(intersection_coords).T, round_value)
    # sort points according to angle
    intersection_angles = np.arctan2(
        transform_to_relative_coords(
            intersection_coords_array[0, :], center_point, axis=0
        ),
        transform_to_relative_coords(
            intersection_coords_array[1, :], center_point, axis=1
        ),
    )
    sort_idx = np.argsort(intersection_angles)
    intersection_coords_array = intersection_coords_array[:, sort_idx]
    # use last point also as first point
    intersection_coords_array = np.concatenate(
        (intersection_coords_array[:, -1][:, np.newaxis], intersection_coords_array),
        axis=1,
    )

    # define distance in x and y between point i and point i+1
    diff_intersection_points = np.abs(np.diff(intersection_coords_array, axis=1))
    # very small differences lead to errors - delete them
    # idx_delete = [i for i in range(diff_intersection_points.shape[1]) if diff_intersection_points.sum(axis=0)[i] < 1e-3]
    # diff_intersection_points = np.delete(diff_intersection_points,idx_delete,axis=1)
    # intersection_coords_array = np.delete(intersection_coords_array,(np.array(idx_delete)+1).tolist(), axis=1)

    # do not use areas where the boundaries are below this threshold - leads to floating errors
    tolerance_boundary = 1e-6

    # if debug_mode:
    rect_list = []

    idx_pixel_grey = []
    color_pixel = []
    for i_diff in range(diff_intersection_points.shape[1]):
        # counter some floating point issues
        # for ceil and floor operations it is needed that value is 2 instead of 1.99999
        # tolerance_float = 1e-3
        # for i_axis in range(2):
        #     for diff_value in [i_diff,i_diff+1]:
        #         if np.abs(intersection_coords_array[i_axis,diff_value] - np.floor(intersection_coords_array[i_axis,diff_value])) < tolerance_float and np.abs(intersection_coords_array[i_axis,diff_value] - np.floor(intersection_coords_array[i_axis,diff_value])) > 0:
        #             intersection_coords_array[i_axis,diff_value] = np.floor(intersection_coords_array[i_axis,diff_value])
        #         if np.abs(intersection_coords_array[i_axis,diff_value] - np.ceil(intersection_coords_array[i_axis,diff_value])) < tolerance_float and np.abs(intersection_coords_array[i_axis,diff_value] - np.ceil(intersection_coords_array[i_axis,diff_value])) > 0:
        #             intersection_coords_array[i_axis,diff_value] = np.ceil(intersection_coords_array[i_axis,diff_value])

        # if distance is 1, use either x or y integration
        # in this way we immediatly get the pixel color value
        if diff_intersection_points[1, i_diff] == 1:
            idx_pixel_x = np.min(
                np.array(
                    [
                        np.floor(intersection_coords_array[0, i_diff]),
                        np.floor(intersection_coords_array[0, i_diff + 1]),
                    ]
                )
            ).astype(int)
            idx_pixel_y = np.min(
                np.array(
                    [
                        np.floor(intersection_coords_array[1, i_diff]),
                        np.floor(intersection_coords_array[1, i_diff + 1]),
                    ]
                )
            ).astype(int)
            if (
                intersection_coords_array[1, i_diff]
                < intersection_coords_array[1, i_diff + 1]
            ):
                lower_boundary_y = intersection_coords_array[1, i_diff]
                upper_boundary_y = intersection_coords_array[1, i_diff + 1]
            else:
                lower_boundary_y = intersection_coords_array[1, i_diff + 1]
                upper_boundary_y = intersection_coords_array[1, i_diff]
            lower_boundary_y_rel = transform_to_relative_coords(
                lower_boundary_y, center_point, axis=1
            )
            upper_boundary_y_rel = transform_to_relative_coords(
                upper_boundary_y, center_point, axis=1
            )
            # integrate over y
            if np.abs(upper_boundary_y_rel - lower_boundary_y_rel) < tolerance_boundary:
                continue
            area = circle_area_segment(
                r, lower_boundary_y_rel, upper_boundary_y_rel, axis=1
            )
            # choose x-coordinate which is the closest to the y-axis
            # x_distance_rel = transform_to_relative_coords(intersection_coords_array[0,i_diff],center_point,axis=0)
            # x_distance_rel_iplus1 = np.abs(transform_to_relative_coords(intersection_coords_array[0,i_diff+1],center_point,axis=0))
            # if x_distance_rel_i < x_distance_rel_iplus1:
            #     x_distance = x_distance_rel_i
            #     idx_pixel_x = np.floor(intersection_coords_array[0,i_diff]).astype(int)
            # else:
            #     x_distance = x_distance_rel_iplus1
            #     idx_pixel_x = np.floor(intersection_coords_array[0,i_diff+1]).astype(int)

            # if i_diff is on grid, choose i_diff+1 for ceil and floor to work
            # if intersection_coords_array[0,i_diff]%1 == 0: # on grid
            #     closer_x_value = intersection_coords_array[0,i_diff+1]
            # else:
            #     closer_x_value = intersection_coords_array[0,i_diff]

            # absolute edge coordinate values
            edge_coordinates_x_abs = np.array([idx_pixel_x, idx_pixel_x + 1])
            # relative egde coordinate values
            edge_coordinates_x_rel = transform_to_relative_coords(
                edge_coordinates_x_abs, center_point, axis=0
            )
            # get inner edge
            inner_edge_x_rel = np.abs(
                edge_coordinates_x_rel[np.argmin(np.abs(edge_coordinates_x_rel))]
            )
            x_distance = inner_edge_x_rel
            # if x_distance_rel < 0:
            #     # left coordinate plane - round up
            #     x_distance = center_point[0] - np.ceil(closer_x_value)
            # else:
            #     # right coordinate plane - round down
            #     x_distance = np.floor(closer_x_value) - center_point[0]
            # assert x_distance >= 0
            area_pixel = area - diff_intersection_points[1, i_diff] * x_distance

            # get pixel index
            # idx_pixel_x = np.floor(intersection_coords_array[0,i_diff]).astype(int)
            idx_pixel_grey.append((idx_pixel_x, idx_pixel_y))
            color_pixel.append(fraction_to_color(area_pixel))
            current_idx = len(idx_pixel_grey) - 1

        elif diff_intersection_points[0, i_diff] == 1:
            # get current pixel (minimum value from both floor coordinates)
            idx_pixel_x = np.min(
                np.array(
                    [
                        np.floor(intersection_coords_array[0, i_diff]),
                        np.floor(intersection_coords_array[0, i_diff + 1]),
                    ]
                )
            ).astype(int)
            idx_pixel_y = np.min(
                np.array(
                    [
                        np.floor(intersection_coords_array[1, i_diff]),
                        np.floor(intersection_coords_array[1, i_diff + 1]),
                    ]
                )
            ).astype(int)

            # integrate over x if x-diff = 1
            if (
                intersection_coords_array[0, i_diff]
                < intersection_coords_array[0, i_diff + 1]
            ):
                lower_boundary_x = intersection_coords_array[0, i_diff]
                upper_boundary_x = intersection_coords_array[0, i_diff + 1]
            else:
                lower_boundary_x = intersection_coords_array[0, i_diff + 1]
                upper_boundary_x = intersection_coords_array[0, i_diff]
            lower_boundary_x_rel = transform_to_relative_coords(
                lower_boundary_x, center_point, axis=0
            )
            upper_boundary_x_rel = transform_to_relative_coords(
                upper_boundary_x, center_point, axis=0
            )
            if np.abs(upper_boundary_x_rel - lower_boundary_x_rel) < tolerance_boundary:
                continue
            area = circle_area_segment(
                r, lower_boundary_x_rel, upper_boundary_x_rel, axis=0
            )
            # choose y-coordinate which is the closest to the x-axis
            y_distance_rel = transform_to_relative_coords(
                intersection_coords_array[1, i_diff], center_point, axis=1
            )
            # if i_diff is on grid, choose i_diff+1 for ceil and floor to work
            if intersection_coords_array[1, i_diff] % 1 == 0:  # on grid
                closer_y_value = intersection_coords_array[1, i_diff + 1]
            else:
                closer_y_value = intersection_coords_array[1, i_diff]
            if y_distance_rel < 0:
                # lower coordinate plane - round up
                y_distance = center_point[1] - np.ceil(closer_y_value)
            else:
                # right coordinate plane - round down
                y_distance = np.floor(closer_y_value) - center_point[1]
            assert y_distance >= 0
            area_pixel = area - diff_intersection_points[0, i_diff] * y_distance

            # idx_pixel_y = np.floor(intersection_coords_array[1,i_diff]).astype(int)
            idx_pixel_grey.append((idx_pixel_x, idx_pixel_y))
            color_pixel.append(fraction_to_color(area_pixel))
            current_idx = len(idx_pixel_grey) - 1

        else:
            # integrate either x or y - here we use also x-integration

            # there are some special cases that need to be detected and considered in the calculation
            # we can perform a normal x-integration if the two intersection points cross the inner or just one outer edge
            # inner edges are the edges closer to the center point, outer edges the remaining two
            # another sub-special case: pixels that cross the x or y-axis of the reference circle

            # 1. get current pixel (minimum value from both floor coordinates)
            idx_pixel_x = np.min(
                np.array(
                    [
                        np.floor(intersection_coords_array[0, i_diff]),
                        np.floor(intersection_coords_array[0, i_diff + 1]),
                    ]
                )
            ).astype(int)
            idx_pixel_y = np.min(
                np.array(
                    [
                        np.floor(intersection_coords_array[1, i_diff]),
                        np.floor(intersection_coords_array[1, i_diff + 1]),
                    ]
                )
            ).astype(int)
            # pixel correction if both pixels lie on same y-grid line and are on bottom/left relative coordinate plane
            if (
                intersection_coords_array[1, i_diff]
                == intersection_coords_array[1, i_diff + 1]
                and intersection_coords_array[1, i_diff] % 1 == 0
            ):
                if (
                    transform_to_relative_coords(
                        intersection_coords_array[1, i_diff], center_point, axis=1
                    )
                    < 0
                ):
                    idx_pixel_y -= 1
            if (
                intersection_coords_array[0, i_diff]
                == intersection_coords_array[0, i_diff + 1]
                and intersection_coords_array[0, i_diff] % 1 == 0
            ):
                if (
                    transform_to_relative_coords(
                        intersection_coords_array[0, i_diff], center_point, axis=0
                    )
                    < 0
                ):
                    idx_pixel_x -= 1
            # 2. get outer edge
            # absolute edge coordinate values
            edge_coordinates_x_abs = np.array([idx_pixel_x, idx_pixel_x + 1])
            edge_coordinates_y_abs = np.array([idx_pixel_y, idx_pixel_y + 1])
            # relative egde coordinate values
            edge_coordinates_x_rel = transform_to_relative_coords(
                edge_coordinates_x_abs, center_point, axis=0
            )
            edge_coordinates_y_rel = transform_to_relative_coords(
                edge_coordinates_y_abs, center_point, axis=1
            )
            if np.prod(edge_coordinates_y_rel) < 0:
                # crosses y border -> switch to y-integration
                y_axis_crossing = True
            else:
                y_axis_crossing = False
            if np.prod(edge_coordinates_x_rel) < 0:
                # crosses any border
                x_axis_crossing = True
            else:
                x_axis_crossing = False
            #  get outer edge depeding on larger relative position
            if y_axis_crossing:
                # choose both y edges as "outer edges"
                outer_edge_y = edge_coordinates_y_abs.tolist()
            else:
                outer_edge_y = [
                    edge_coordinates_y_abs[np.argmax(np.abs(edge_coordinates_y_rel))]
                ]
            if x_axis_crossing:
                # choose both x edges as "outer edges"
                outer_edge_x = edge_coordinates_x_abs.tolist()
            else:
                outer_edge_x = [
                    edge_coordinates_x_abs[np.argmax(np.abs(edge_coordinates_x_rel))]
                ]

            # check if both points lie on outer edges
            # first get outer values
            intersect_x_values_abs = np.array(
                [
                    intersection_coords_array[0, i_diff],
                    intersection_coords_array[0, i_diff + 1],
                ]
            )
            intersect_x_values_rel = transform_to_relative_coords(
                intersect_x_values_abs, center_point, axis=0
            )
            outer_value_x = intersect_x_values_abs[
                np.argmax(np.abs(intersect_x_values_rel))
            ]
            intersect_y_values_abs = np.array(
                [
                    intersection_coords_array[1, i_diff],
                    intersection_coords_array[1, i_diff + 1],
                ]
            )
            intersect_y_values_rel = transform_to_relative_coords(
                intersect_y_values_abs, center_point, axis=1
            )
            # check if outer value of y comes from i_diff (0) or i_diff+1 (1)
            idx_outer_value_y = np.argmax(np.abs(intersect_y_values_rel))
            outer_value_y = intersect_y_values_abs[idx_outer_value_y]

            # initialize/reset rectangle_area
            rectangle_area = 0
            if outer_value_x in outer_edge_x and outer_value_y in outer_edge_y:
                # adding the missing rectangle of the pixel is necessary
                if not y_axis_crossing:
                    rectangle_distance = 1 - np.abs(
                        intersection_coords_array[0, i_diff + 1]
                        - intersection_coords_array[0, i_diff]
                    )
                else:
                    rectangle_distance = 1 - np.abs(
                        intersection_coords_array[1, i_diff + 1]
                        - intersection_coords_array[1, i_diff]
                    )

                # rectangle_area = rectangle_distance since pixel height is 1
                rectangle_area = rectangle_distance
                # print('outer_edge crossing detected rectangle area added')

            if not y_axis_crossing:
                if (
                    intersection_coords_array[0, i_diff]
                    < intersection_coords_array[0, i_diff + 1]
                ):
                    lower_boundary_x = intersection_coords_array[0, i_diff]
                    upper_boundary_x = intersection_coords_array[0, i_diff + 1]
                else:
                    lower_boundary_x = intersection_coords_array[0, i_diff + 1]
                    upper_boundary_x = intersection_coords_array[0, i_diff]
                lower_boundary_x_rel = transform_to_relative_coords(
                    lower_boundary_x, center_point, axis=0
                )
                upper_boundary_x_rel = transform_to_relative_coords(
                    upper_boundary_x, center_point, axis=0
                )
                if (
                    np.abs(upper_boundary_x_rel - lower_boundary_x_rel)
                    < tolerance_boundary
                ):
                    continue
                area = circle_area_segment(
                    r, lower_boundary_x_rel, upper_boundary_x_rel, axis=0
                )
                # choose y-coordinate which is the closest to the x-axis of reference circle

                # idx_inner_value_y = np.argmin(np.abs(intersect_y_values_rel))
                # get inner edge
                inner_edge_y_rel = np.abs(
                    edge_coordinates_y_rel[np.argmin(np.abs(edge_coordinates_y_rel))]
                )
                y_distance = inner_edge_y_rel
            else:
                # y-integration due to relative y-axis border crossing
                # print('y-integration is used')
                if (
                    intersection_coords_array[1, i_diff]
                    < intersection_coords_array[1, i_diff + 1]
                ):
                    lower_boundary_y = intersection_coords_array[1, i_diff]
                    upper_boundary_y = intersection_coords_array[1, i_diff + 1]
                else:
                    lower_boundary_y = intersection_coords_array[1, i_diff + 1]
                    upper_boundary_y = intersection_coords_array[1, i_diff]
                lower_boundary_y_rel = transform_to_relative_coords(
                    lower_boundary_y, center_point, axis=1
                )
                upper_boundary_y_rel = transform_to_relative_coords(
                    upper_boundary_y, center_point, axis=1
                )
                # integrate over y
                if (
                    np.abs(upper_boundary_y_rel - lower_boundary_y_rel)
                    < tolerance_boundary
                ):
                    continue
                area = circle_area_segment(
                    r, lower_boundary_y_rel, upper_boundary_y_rel, axis=1
                )
                # idx_inner_value_x = np.argmin(np.abs(intersect_x_values_rel))
                # get inner edge
                inner_edge_x_rel = np.abs(
                    edge_coordinates_x_rel[np.argmin(np.abs(edge_coordinates_x_rel))]
                )
                x_distance = inner_edge_x_rel

            # special case x-border is crossed - multiple areas within pixel
            # only happens if diff < 1 therefore only implemented for this case
            current_pixel = (idx_pixel_x, idx_pixel_y)
            if current_pixel in idx_pixel_grey:
                idx_doubling_grey = idx_pixel_grey.index(current_pixel)
                # reset rectangle area since all parts are calculated separately
                rectangle_area = 0
                if not y_axis_crossing:
                    area_pixel = (
                        area
                        - diff_intersection_points[0, i_diff] * y_distance
                        + rectangle_area
                    )
                else:
                    area_pixel = (
                        area
                        - diff_intersection_points[1, i_diff] * x_distance
                        + rectangle_area
                    )

                if color_pixel[idx_doubling_grey] < fraction_to_color(
                    0.5
                ):  # mostly black
                    # rectangle has already been added first delete the addtional area the current intersection as rectangle
                    if not y_axis_crossing:
                        color_pixel[idx_doubling_grey] += fraction_to_color(
                            diff_intersection_points[0, i_diff], only_white_part=True
                        )
                    else:
                        color_pixel[idx_doubling_grey] += fraction_to_color(
                            diff_intersection_points[1, i_diff], only_white_part=True
                        )
                area_pixel = np.around(area_pixel, round_value)
                color_pixel[idx_doubling_grey] -= fraction_to_color(
                    area_pixel, only_white_part=True
                )
                current_idx = idx_doubling_grey
                # print(f'Double pixel detected. Add multiple areas instead of rectangle area.')
                assert color_pixel[idx_doubling_grey] > 0
            else:
                if not y_axis_crossing:
                    area_pixel = (
                        area
                        - diff_intersection_points[0, i_diff] * y_distance
                        + rectangle_area
                    )
                else:
                    area_pixel = (
                        area
                        - diff_intersection_points[1, i_diff] * x_distance
                        + rectangle_area
                    )
                area_pixel = np.around(area_pixel, round_value)
                assert area_pixel >= -tolerance_boundary and area_pixel <= 1

                if area_pixel < tolerance_boundary:
                    continue
                idx_pixel_grey.append((idx_pixel_x, idx_pixel_y))
                color_pixel.append(fraction_to_color(area_pixel))
                current_idx = len(idx_pixel_grey) - 1

        # scatter plot
        # if debug_mode:
        rect_list.append(
            plt.Rectangle(
                idx_pixel_grey[current_idx],
                1,
                1,
                color=(
                    color_pixel[current_idx] / 255,
                    color_pixel[current_idx] / 255,
                    color_pixel[current_idx] / 255,
                ),
            )
        )

    # get black indices (fill gap between grey index values)
    sorted_values = np.array(sorted(idx_pixel_grey))
    x_indices_grey = np.unique(sorted_values[:, 0])
    idx_pixel_black = []
    for i_x_idx in x_indices_grey:
        # all indices of row x
        indices_i = sorted_values[np.where(sorted_values[:, 0] == i_x_idx)]
        if indices_i.shape[0] == 1:
            # if there is just a single grey pixel in this line, we cannot calculate the difference
            # no black pixel in this x row
            continue
        diff_indices_i = np.diff(indices_i[:, 1])
        if np.max(diff_indices_i) <= 1:
            # no black pixels in this x-column
            continue
        else:
            # find gap that needs to be filled in y-axis with black pixels
            gap_idx = np.where(diff_indices_i > 1)
            assert len(gap_idx) == 1
            for black_idx_y in range(
                int(indices_i[gap_idx[0], 1] + 1), int(indices_i[gap_idx[0] + 1, 1])
            ):
                idx_pixel_black.append((i_x_idx, black_idx_y))

    if debug_mode:
        fig, ax = plt.subplots()
        for patch_rect in rect_list:
            # workaround for RuntimeError: "Can not put single artist in more than one figure"
            copy_patch_rect = copy(patch_rect)
            ax.add_patch(copy_patch_rect)
        for idx_black in idx_pixel_black:
            ax.add_patch(plt.Rectangle(idx_black, 1, 1, color="k"))

        plt.scatter(intersection_coords_array[0, :], intersection_coords_array[1, :])
        circle1 = plt.Circle(center_point, radius=r, fill=False)
        ax.add_patch(circle1)

        spacing = 1
        loc = plticker.MultipleLocator(spacing)
        ax.xaxis.set_major_locator(loc)
        ax.yaxis.set_major_locator(loc)
        plt.grid(visible=True, which="major")

        ax.set_xlim((-2 * r + center_point[0], 2 * r + center_point[0]))
        ax.set_ylim((-2 * r + center_point[1], 2 * r + center_point[1]))
        ax.set_aspect("equal", adjustable="box")

    pixel_array[tuple(np.array(idx_pixel_grey).T)] = np.array(color_pixel)
    if len(idx_pixel_black) != 0:
        pixel_array[tuple(np.array(idx_pixel_black).T)] = 0

    if debug_mode:
        data = Image.fromarray(pixel_array.T).convert("L")
        # saving the final output as a PNG file
        data.save("pixel_array_circle_debug.png")
        print("debug stop")

    return pixel_array


# %% create rectangle shape
def create_pixel_rectangle(center_point, pixel_array, mass_pixel_size=None):
    """
    Create an pixel array
    :param x: array(n_t*n_s,n): trajectories of n/2 masses
    :param time_idx: int: time index of requested video frame
    :param num_pixels: array (2,): number of pixels in each direction {[100 100]}
    :param mass_pixel_size: array (2,): size in pixels of one mass
    """

    raise NotImplementedError(f"The rectangle version needs to be reworked.")

    pixel_value_black = 0

    # default values
    if mass_pixel_size is None:
        mass_pixel_size = np.array([1, 1])

    # pixel_array = np.random.rand(num_pixels[0],num_pixels[1])*256

    # # check if scale factor is too high
    # # check minimal value of first mass and maximum value of last mass
    # min_first_mass = np.min(x[:,0])
    # max_last_mass = np.max(x[:,n_mass-1])
    # most_left_value = pos_x_center[0] - min_first_mass - np.ceil(mass_pixel_size[0]/2)
    # most_right_value = pos_x_center[n_mass-1] + max_last_mass + np.ceil(mass_pixel_size[0]/2)

    # if most_left_value < 0 or most_right_value > num_pixels[0]:
    #     raise ValueError('Masses will cross the image border. Please increase the number of pixel in x or decrease the pixel scale factor.')

    pos_x_max = center_point[0] + mass_pixel_size[0] / 2
    pos_x_min = center_point[0] - mass_pixel_size[0] / 2
    pos_y_max = center_point[1] + mass_pixel_size[1] / 2
    pos_y_min = center_point[1] - mass_pixel_size[1] / 2

    # %% define all black pixels
    lower_edge_black_x = np.ceil(pos_x_min).astype(int)
    lower_edge_black_y = np.ceil(pos_y_min).astype(int)
    # floor -1 due to 0 indexing
    upper_edge_black_x = np.floor(pos_x_max).astype(int) - 1
    upper_edge_black_y = np.floor(pos_y_max).astype(int) - 1

    # get all indices for black pixels TODO: can probably be shortened
    def generate_index_combinations(start_indices, end_indices):
        index_combinations = list(itertools.product(start_indices, end_indices))
        return index_combinations

    # create list of index tuples
    idx_black = generate_index_combinations(
        [*range(lower_edge_black_x, upper_edge_black_x + 1)],
        [*range(lower_edge_black_y, upper_edge_black_y + 1)],
    )

    # flatten list
    idx_black = [item for sublist in idx_black for item in sublist]

    # feed to pixel array but convert to two tupels for x and y
    pixel_array[tuple(np.array(idx_black).T)] = pixel_value_black

    # %% define gray pixels
    # left grey edge
    idx_left_start_x = lower_edge_black_x - 1
    idx_left_start_y = lower_edge_black_y
    idx_left_end_x = lower_edge_black_x - 1
    idx_left_end_y = upper_edge_black_y
    distance_left_edge = np.zeros(n_mass)
    for i in range(n_mass):
        idx_comb_i = generate_index_combinations(
            [*range(idx_left_start_x[i], idx_left_end_x[i] + 1)],
            [*range(idx_left_start_y[i], idx_left_end_y[i] + 1)],
        )
        distance_left_edge[i] = np.ceil(pos_x_min[i]) - pos_x_min[i]
        pixel_value_grey_left = 256 - distance_left_edge[i] * 256
        pixel_array[tuple(np.array(idx_comb_i).T)] = pixel_value_grey_left

    # right grey edge
    idx_right_start_x = upper_edge_black_x + 1
    idx_right_start_y = lower_edge_black_y
    idx_right_end_x = upper_edge_black_x + 1
    idx_right_end_y = upper_edge_black_y
    distance_right_edge = np.zeros(n_mass)
    for i in range(n_mass):
        idx_comb_i = generate_index_combinations(
            [*range(idx_right_start_x[i], idx_right_end_x[i] + 1)],
            [*range(idx_right_start_y[i], idx_right_end_y[i] + 1)],
        )
        distance_right_edge[i] = pos_x_max[i] - np.floor(pos_x_max[i])
        pixel_value_grey_right = 256 - distance_right_edge[i] * 256
        pixel_array[tuple(np.array(idx_comb_i).T)] = pixel_value_grey_right

    # bottom grey edge
    idx_bottom_start_x = lower_edge_black_x
    idx_bottom_start_y = lower_edge_black_y - 1
    idx_bottom_end_x = upper_edge_black_x
    idx_bottom_end_y = lower_edge_black_y - 1
    distance_bottom_edge = np.zeros(n_mass)
    for i in range(n_mass):
        idx_comb_i = generate_index_combinations(
            [*range(idx_bottom_start_x[i], idx_bottom_end_x[i] + 1)],
            [*range(idx_bottom_start_y[i], idx_bottom_end_y[i] + 1)],
        )
        distance_bottom_edge[i] = np.ceil(pos_y_min[i]) - pos_y_min[i]
        pixel_value_grey_bottom = 256 - distance_bottom_edge[i] * 256
        pixel_array[tuple(np.array(idx_comb_i).T)] = pixel_value_grey_bottom

    # top grey edge
    idx_top_start_x = lower_edge_black_x
    idx_top_start_y = upper_edge_black_y + 1
    idx_top_end_x = upper_edge_black_x
    idx_top_end_y = upper_edge_black_y + 1
    distance_top_edge = np.zeros(n_mass)
    for i in range(n_mass):
        idx_comb_i = generate_index_combinations(
            [*range(idx_top_start_x[i], idx_top_end_x[i] + 1)],
            [*range(idx_top_start_y[i], idx_top_end_y[i] + 1)],
        )
        distance_top_edge[i] = pos_y_max[i] - np.floor(pos_y_max[i])
        pixel_value_grey_top = 256 - distance_top_edge[i] * 256
        pixel_array[tuple(np.array(idx_comb_i).T)] = pixel_value_grey_top

    # corner values
    for i in range(n_mass):
        # calculate corner areas and divide by pixel area which is 1
        top_left_corner_area = distance_left_edge[i] * distance_top_edge[i]
        top_right_corner_area = distance_right_edge[i] * distance_top_edge[i]
        bottom_left_corner_area = distance_bottom_edge[i] * distance_left_edge[i]
        bottom_right_corner_area = distance_bottom_edge[i] * distance_right_edge[i]
        # define indices
        index_top_left = (lower_edge_black_x[i] - 1, upper_edge_black_y[i] + 1)
        index_top_right = (upper_edge_black_x[i] + 1, upper_edge_black_y[i] + 1)
        index_bottom_left = (lower_edge_black_x[i] - 1, lower_edge_black_y[i] - 1)
        index_bottom_right = (upper_edge_black_x[i] + 1, lower_edge_black_y[i] - 1)
        # write to pixel array
        pixel_array[index_top_left] = 256 - top_left_corner_area * 256
        pixel_array[index_top_right] = 256 - top_right_corner_area * 256
        pixel_array[index_bottom_left] = 256 - bottom_left_corner_area * 256
        pixel_array[index_bottom_right] = 256 - bottom_right_corner_area * 256

    # transpose for mixing x-y and row-column
    return pixel_array.T
