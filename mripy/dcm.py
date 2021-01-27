#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from scipy import stats


# Visualization
# =============
from matplotlib import pyplot as plt
from matplotlib.path import Path
from matplotlib import patches
from matplotlib.colors import Normalize

# Custom BoxStyle for usage in annotate(bbox=dict(boxstyle='ext'))
class ExtendedCircle(patches.BoxStyle.Circle):
    '''
    An extended Circle BoxStyle that keeps a minimum predefined width parameter.

    References
    ----------
    https://stackoverflow.com/questions/40796117/how-do-i-make-the-width-of-the-title-box-span-the-entire-plot       
    https://github.com/matplotlib/matplotlib/blob/master/lib/matplotlib/patches.py
    '''

    def __init__(self, width, pad=0.3):
        '''
        width : 
            Predefined width of the bbox. 
        pad : 
            The amount of padding around the original bbox.
        '''
        self.width = width
        super().__init__(pad=pad)

    def transmute(self, x0, y0, width, height, mutation_size):
    # def __call__(self, x0, y0, width, height, mutation_size):
        '''
        x0 and y0 are the lower left corner of original bbox.
        They are set automatically by matplotlib.
        '''
        xc, yc = x0 + width/2, y0 + height/2 # Center of the circle
        pad = mutation_size * self.pad
        radius = max(self.width/2, width/2+pad)
        return Path.circle((xc, yc), radius) # Return a Path object as the new bbox

# Register the custom BoxStyle
patches.BoxStyle._style_list['ext_circle'] = ExtendedCircle


class ExtendedSimple(patches.ArrowStyle._Base):
    """A simple arrow. Only works with a quadratic Bezier curve."""

    def __init__(self, head_length=.5, head_width=.5, tail_width=.2):
        """
        Parameters
        ----------
        head_length : float, default: 0.5
            Length of the arrow head.
        head_width : float, default: 0.5
            Width of the arrow head.
        tail_width : float, default: 0.2
            Width of the arrow tail.
        """
        self.head_length, self.head_width, self.tail_width = \
            head_length, head_width, tail_width
        super().__init__()

    def transmute(self, path, mutation_size, linewidth):

        x0, y0, x1, y1, x2, y2 = self.ensure_quadratic_bezier(path)

        # divide the path into a head and a tail
        head_length = self.head_length * mutation_size
        in_f = patches.inside_circle(x2, y2, head_length)
        arrow_path = [(x0, y0), (x1, y1), (x2, y2)]

        arrow_out, arrow_in = \
            patches.split_bezier_intersecting_with_closedpath(
                arrow_path, in_f, tolerance=0.01)

        # head
        head_width = self.head_width * mutation_size
        head_left, head_right = patches.make_wedged_bezier2(arrow_in,
                                                    head_width / 2., wm=.5)

        # tail
        tail_width = self.tail_width * mutation_size
        tail_left, tail_right = patches.get_parallels(arrow_out,
                                                tail_width / 2.)

        patch_path = [(Path.MOVETO, tail_right[0]),
                        (Path.CURVE3, tail_right[1]),
                        (Path.CURVE3, tail_right[2]),
                        (Path.LINETO, head_right[0]),
                        (Path.CURVE3, head_right[1]),
                        (Path.CURVE3, head_right[2]),
                        (Path.CURVE3, head_left[1]),
                        (Path.CURVE3, head_left[0]),
                        (Path.LINETO, tail_left[2]),
                        (Path.CURVE3, tail_left[1]),
                        (Path.CURVE3, tail_left[0]),
                        (Path.LINETO, tail_right[0]),
                        (Path.CLOSEPOLY, tail_right[0]),
                        ]

        path = Path([p for c, p in patch_path], [c for c, p in patch_path])

        return path, True

# Register the custom ArrowStyle
patches.ArrowStyle._style_list['ext_simple'] = ExtendedSimple



def draw_circle_arrow(xy, radius, patchA=None, ax=None):
    '''
    References
    ----------
    https://stackoverflow.com/questions/37512502/how-to-make-arrow-that-loops-in-matplotlib/38224201
    '''
    # Arrow line
    arc = patches.Arc(xy, radius,radius,angle=angle_,
          theta1=0,theta2=theta2_,capstyle='round',linestyle='-',lw=10,color=color_)
    ax.add_patch(arc)


    # # Arrow head
    # endX=centX+(radius/2)*np.cos(rad(theta2_+angle_)) #Do trig to determine end position
    # endY=centY+(radius/2)*np.sin(rad(theta2_+angle_))

    # ax.add_patch(                    #Create triangle as arrow head
    #     RegularPolygon(
    #         (endX, endY),            # (x,y)
    #         3,                       # number of vertices
    #         radius/9,                # radius
    #         rad(angle_+theta2_),     # orientation
    #         color=color_
    #     )
    # )




def plot_dcm(rois, inputs, A, B, C, PA=None, PB=None, PC=None, roi_order=None, rads=None, ax=None):
    '''
    Visualize a DCM (as well as estimated A, B, C parameters).

    Parameters
    ----------
    A : [n_rois, n_rois]
        Baseline connectivity, dest <- src
    B : [n_rois, n_rois, n_inputs]
        Modulation on connectivity (non-zero for modulatory inputs only)
    C : [n_rois, n_inputs]
        Driving on nodes (non-zero for driving inputs only)
        
    References
    ----------
    https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.annotate.html
    https://matplotlib.org/stable/gallery/userdemo/annotate_simple_coord01.html
    
    https://matplotlib.org/3.3.4/gallery/lines_bars_and_markers/gradient_bar.html (gradient fill)
    '''
    n_rois = len(rois)
    n_inputs = len(inputs)
    if rads is None:
        rads = np.nan*np.ones([n_rois, n_rois])
    roi_ids = np.arange(n_rois)
    if roi_order is not None:
        rois = rois[roi_order]
        if A is not None:
            A = A[roi_order][:,roi_order]
        if B is not None:
            B = B[roi_order][:,roi_order,:]
        if C is not None:
            C = C[roi_order]
        if PA is not None:
            PA = PA[roi_order][:,roi_order]
        if PB is not None:
            PB = PB[roi_order][:,roi_order,:]
        if PC is not None:
            PC = PC[roi_order]
        rads = rads[roi_order][:,roi_order]
        roi_ids = roi_ids[roi_order]
    # Prepare axes
    if ax is None:
        ax = plt.gca()
    plt.axis('off')
    # Nodes layout (circular)
    center = [0.5, 0.45]
    radius = 0.4
    roi_angles = np.pi/2 - 2*np.pi*(np.arange(n_rois) + 0.5*(1-n_rois%2))/n_rois
    roi_xys = center + radius*np.c_[np.cos(roi_angles), np.sin(roi_angles)]
    roi_hands = (np.sin(-roi_angles + np.pi/2) >= 0)*2-1 # center/right=1, left=-1
    # Draw nodes
    roi_patches = []
    for k in range(n_rois):
        annot = ax.annotate(f"$^{roi_ids[k]}\\ $\n{rois[k]}" if False else rois[k], 
            xy=roi_xys[k], xycoords='axes fraction',
            ha='center', va='center', color='k',
            bbox=dict(boxstyle='ext_circle,width=70,pad=0.3', ec='k', fc='w'))
        roi_patches.append(annot.get_bbox_patch())
    # Normalize edges
    norms = {k: Normalize(vmin=0, vmax=np.abs(E).max()) for k, E in zip(['A', 'B', 'C'], [A, B, C]) if E is not None}
    norms['G'] = Normalize(vmin=0, vmax=np.abs(np.concatenate([E.ravel() for E in [A, B, C] if E is not None]).max())) # Global normalization over A, B, C
    # Draw connections
    def draw_connections(E, P=None, norm=norms['G']):
        '''
        E : parameter value (can be either A or B(i))
        P : posterior probability (significance)
        '''
        S = P > 0.95 if P is not None else np.ones_like(E, dtype=bool)
        for dst in range(n_rois):
            for src in range(n_rois):
                if E[dst,src] != 0:
                    if dst == src: # Self connections
                        # ax.annotate('', xy=roi_xys[dst], xycoords='axes fraction',
                        #     xytext=roi_xys[src], textcoords='axes fraction',
                        #     arrowprops=dict(
                        #         patchA=roi_patches[src], patchB=roi_patches[dst],
                        #         arrowstyle=f"simple,tail_width={max(0.1, norm(E[dst,src]))},\
                        #             head_width={max(0.5, norm(E[dst,src])*2)},head_length=1.0", 
                        #         connectionstyle=f"arc3,rad={(-2 if np.isnan(rads[dst,src]) else rads[dst,src])}",
                        #         ec='w', fc='b', alpha=(1 if S[dst,src] else 0.1),
                        #     ))
                        
                        
                        # annot = ax.annotate('haha', 
                        #     xy=roi_xys[dst], xycoords='axes fraction',
                        #     ha='center', va='center', color='k',
                        #     bbox=dict(boxstyle='ext_circle,width=70,pad=0.3', 
                        #         ec=('r' if E[dst,src]>0 else 'b'), fc='none', lw=max(0.1, norm(abs(E[dst,src])))))
                        # print(E[dst,src])
                        pass
                    else: # Between area connections
                        if (n_rois%2==1 and dst+src==n_rois) or (n_rois%2==0 and dst+src==n_rois-1):
                            rad_sign = 1
                        elif roi_hands[dst] < 0:
                            rad_sign = -1 # Flip edge curvature if 
                        else:
                            rad_sign = 1
                        ax.annotate('', xy=roi_xys[dst], xycoords='axes fraction',
                            xytext=roi_xys[src], textcoords='axes fraction',
                            arrowprops=dict(
                                patchA=roi_patches[src], patchB=roi_patches[dst],
                                arrowstyle=f"simple,tail_width={max(0.1, norm(abs(E[dst,src])))},\
                                    head_width={max(0.5, norm(abs(E[dst,src]))*2)},head_length=1.0", 
                                connectionstyle=f"arc3,rad={(rad_sign*0.2 if np.isnan(rads[dst,src]) else rads[dst,src])}",
                                ec='w', fc=('r' if E[dst,src]>0 else 'b'), alpha=(1 if S[dst,src] else 0.1),
                            ))

                    # # Arrow head
                    # ax.annotate('', xy=roi_xys[dst], xycoords='axes fraction',
                    #     xytext=roi_xys[src], textcoords='axes fraction',
                    #     arrowprops=dict(
                    #         patchA=roi_patches[src], patchB=roi_patches[dst],
                    #         arrowstyle=f"simple,tail_width=0,head_width=0.7,head_length=1.4", 
                    #         connectionstyle=f"arc3,rad={(0.2 if np.isnan(rads[dst,src]) else rads[dst,src])}",
                    #         ec='w', fc='r', alpha=(1 if P[dst,src]>0.95 else 0.1),
                    #     ))
                    # # Arrow line
                    # ax.annotate('', xy=roi_xys[dst], xycoords='axes fraction',
                    #     xytext=roi_xys[src], textcoords='axes fraction',
                    #     arrowprops=dict(
                    #         patchA=roi_patches[src], patchB=roi_patches[dst],
                    #         arrowstyle=f"-", shrinkB=20, capstyle='butt',
                    #         connectionstyle=f"arc3,rad={(0.2 if np.isnan(rads[dst,src]) else rads[dst,src])}",
                    #         ec='r', fc='g', alpha=(1 if S[dst,src] else 0.1), ls=('-' if S[dst,src] else '--'), lw=abs(E[dst,src])*20
                    #     ))
    draw_connections(A, PA)
    # draw_connections(B[...,2], PB[...,2])
    # draw_connections(B[...,3], PB[...,3])

    # Draw inputs
    if C is not None:
        S = PC > 0.95 if PC is not None else np.ones_like(C, dtype=bool)
        norm = norms['G']
        d_vec = lambda v, c, rot=0: (v-c)/np.linalg.norm(v-c) @ np.array([[np.cos(rot), np.sin(rot)], [-np.sin(rot), np.cos(rot)]])
        for k in range(n_rois):
            input_num = np.sum(C[k]!=0)
            for n in range(n_inputs):
                if C[k,n] != 0:
                    ax.annotate('', xy=roi_xys[k], xycoords='axes fraction',
                        xytext=roi_xys[k]+d_vec(roi_xys[k], center, -np.pi/6*roi_hands[k])*0.25, textcoords='axes fraction',
                        arrowprops=dict(
                            patchB=roi_patches[k],
                            arrowstyle=f"simple,tail_width={max(0.1, norm(abs(C[k,n])))},\
                                head_width={max(0.5, norm(abs(C[k,n]))*2)},head_length=1.0", 
                            connectionstyle=f"arc3,rad=0",
                            fc='w', ec=('r' if C[k,n]>0 else 'b'), alpha=(1 if S[k,n] else 0.1),
                        ))
                    # ax.annotate('', xy=roi_xys[k], xycoords='axes fraction',
                    #     xytext=roi_xys[k]+d_vec(roi_xys[k], center)*0.25, textcoords='axes fraction',
                    #     arrowprops=dict(
                    #         patchB=roi_patches[k],
                    #         arrowstyle=f"simple,tail_width={max(0.1, norm(abs(C[k,n])))/2},\
                    #             head_width={max(0.5, norm(abs(C[k,n]))*2)/2},head_length=1.0", 
                    #         connectionstyle=f"arc3,rad=0",
                    #         ec='none', fc=('w' if C[k,n]>0 else 'b'), alpha=(1 if S[k,n] else 0.1),
                    #     ))
                    # ax.annotate('', xy=roi_xys[k], xycoords='axes fraction',
                    #     xytext=roi_xys[k]+d_vec(roi_xys[k], center)*0.25, textcoords='axes fraction',
                    #     arrowprops=dict(
                    #         patchB=roi_patches[k],
                    #         arrowstyle=f"-", 
                    #         connectionstyle=f"arc3,rad=0",
                    #         ec='w', alpha=(1 if S[k,n] else 0.1),
                    #     ))

def plot_peb_bma(GCM, **kwargs):
    # Parse relevant variables from Matlab struct
    n_subjects = GCM.shape[0]
    DCM = GCM[0,0]
    xY = DCM['xY'][0,0]
    rois = np.array([x[0] for x in xY['name'].flat]) # Potential bug: Sess
    U = DCM['U'][0,0]
    inputs = np.array([x[0] for x in U['name'][0,0].flat])
    a, b, c = (DCM[x][0,0] for x in ['a', 'b', 'c'])
    As, Bs, Cs = (np.array([GCM[k,0]['Ep'][0,0][x][0,0] for k in range(n_subjects)]) for x in ['A', 'B', 'C'])
    Ag, Bg, Cg = As.mean(0), Bs.mean(0), Cs.mean(0) # Simple group average
    PAg, PBg, PCg = (1-stats.ttest_1samp(Es, 0, axis=0).pvalue for Es in [As, Bs, Cs])

    # Plot DCM
    fig, axs = plt.subplots(2, 2, figsize=[15,15])
    plt.sca(axs[0,0])
    plot_dcm(rois, inputs, Ag, Bg, Cg, PA=PAg, PB=PBg, PC=PCg, **kwargs)
    plt.sca(axs[0,1])
    plt.axis('off')
    plt.sca(axs[1,0])
    plot_dcm(rois, inputs, Bg[...,2], Bg, Cg, PA=PAg, PB=PBg, PC=PCg, **kwargs)
    plt.sca(axs[1,1])
    plot_dcm(rois, inputs, Bg[...,3], Bg, Cg, PA=PAg, PB=PBg, PC=PCg, **kwargs)
