import re

import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, TextBox
from sympy import *


def widgvis(fig):
    """Hides the toolbar, header, and footer of a figure."""
    fig.canvas.toolbar_visible = False
    fig.canvas.header_visible = False
    fig.canvas.footer_visible = False


def between(a, b, x):
    """Determines if a point x is between a and b. a may be greater or less than b.

    Parameters
    ----------
    a : float
        The first boundary.
    b : float
        The second boundary.
    x : float
        The point to check.

    Returns
    -------
    bool
        True if x is between a and b, False otherwise.
    """
    if a > b:
        return b <= x <= a
    if b > a:
        return a <= x <= b


def near(pt, alist, dist=15):
    """Checks if a point is near any point in a list.

    Parameters
    ----------
    pt : tuple
        The x, y coordinates of the point.
    alist : list
        The list of points to check.
    dist : float, optional
        The distance to consider as "near", by default 15.

    Returns
    -------
    tuple
        A tuple (bool, object), where the bool is True if the point is near any point in the list, and the object is the point in the list that is near. If the point is not near any point in the list, returns (False, None).
    """
    for a in alist:
        x, y = (
            a.ao.get_position()
        )  # Use (bot left, bot right) data coords, not relative
        x = x - 5
        y = y + 2.5
        if 0 < (pt[0] - x) < 25 and 0 < (y - pt[1]) < 25:
            return (True, a)
    return (False, None)


def inboxes(pt, boxlist):
    """Checks if a point is within any box in a list.

    Parameters
    ----------
    pt : tuple
        The x, y coordinates of the point.
    boxlist : list
        The list of boxes to check.

    Returns
    -------
    tuple
        A tuple (bool, object), where the bool is True if the point is within any box in the list, and the object is the box in the list that contains the point. If the point is not within any box in the list, returns (False, None).
    """
    for b in boxlist:
        if b.inbox(pt):
            return (True, b)
    return (False, None)


class avalue:
    """One of the values on the figure that can be filled in."""

    def __init__(self, value, pt, cl):
        """Initializes the AValue.

        Parameters
        ----------
        value : float
            The value.
        pt : tuple
            The x, y coordinates of the value.
        cl : str
            The color of the value.
        """
        self.value = value
        self.cl = cl  # color
        self.pt = pt  # point

    def add_anote(self, ax):
        """Adds an annotation to the value.

        Parameters
        ----------
        ax : Axes
            The axes for the annotation.
        """
        self.ax = ax
        self.ao = self.ax.annotate("?", self.pt, c=self.cl, fontsize="x-small")


class astring:
    """A string that can be set visible or invisible."""

    def __init__(self, ax, string, pt, cl):
        """Initializes the AString.

        Parameters
        ----------
        ax : Axes
            The axes for the string.
        string : str
            The string.
        pt : tuple
            The x, y coordinates of the string.
        cl : str
            The color of the string.
        """
        self.string = string
        self.cl = cl  # color
        self.pt = pt  # point
        self.ax = ax
        self.ao = self.ax.annotate(self.string, self.pt, c="white", fontsize="x-small")

    def astring_visible(self):
        """Makes the string visible."""
        self.ao.set_color(self.cl)

    def astring_invisible(self):
        """Makes the string invisible."""
        self.ao.set_color("white")


class ABox:
    """One of the boxes in the graph that has a value."""

    def __init__(self, ax, value, left, bottom, right, top, anpt, cl, adj_anote_obj):
        """Initializes the ABox.

        Parameters
        ----------
        ax : Axes
            The axes for the box.
        value : float
            The correct value for annotation.
        left : float
            The left coordinate of the box.
        bottom : float
            The bottom coordinate of the box.
        right : float
            The right coordinate of the box.
        top : float
            The top coordinate of the box.
        anpt : tuple
            The x, y coordinates where expression should be listed.
        cl : str
            The color of the box.
        adj_anote_obj : object
            The secondary text for marking edges or None.
        """
        self.ax = ax
        self.value = value  # Correct value for annotation
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.anpt = anpt  # X, y where expression should be listed
        self.cl = cl
        self.ao = self.ax.annotate("?", self.anpt, c=self.cl, fontsize="x-small")
        self.astr = adj_anote_obj  # Secondary text for marking edges or None

    def inbox(self, pt):
        """Checks if a point is within the box.

        Parameters
        ----------
        pt : tuple
            The x, y coordinates of the point.

        Returns
        -------
        bool
            True if the point is within the box, False otherwise.
        """
        x, y = pt
        isbetween = between(self.top, self.bottom, y) and between(
            self.left, self.right, x
        )
        return isbetween

    def update_val(self, value, cl=None):
        """Updates the value of the box.

        Parameters
        ----------
        value : float
            The new value.
        cl : str, optional
            The color of the box.
        """
        self.ao.set_text(value)
        if cl:
            self.ao.set_c(cl)
        else:
            self.ao.set_c(self.cl)

    def show_secondary(self):
        """Shows the secondary text."""
        if self.astr:  # If there is a secondary set of text
            self.astr.ao.set_c("green")

    def clear_secondary(self):
        """Clears the secondary text."""
        if self.astr:  # If there is a secondary set of text
            self.astr.ao.set_c("white")


class plt_network:
    """Handles the plotting of a network."""

    def __init__(self, fn, image, out=None):
        """Initializes the PltNetwork.

        Parameters
        ----------
        fn : function
            The function to call to configure the network.
        image : str
            The path to the image file.
        out : output, optional
            The output for debugging.
        """
        self.out = out  # Debug
        img = plt.imread(image)
        self.fig, self.ax = plt.subplots(figsize=self.sizefig(img))
        boxes = fn(self.ax)
        self.boxes = boxes
        widgvis(self.fig)
        self.ax.xaxis.set_visible(False)
        self.ax.yaxis.set_visible(False)
        self.ax.imshow(img)
        self.fig.text(0.1, 0.9, "Click in boxes to fill in values.")
        self.glist = []  # Place to stash global things
        self.san = []  # Selected annotation

        self.cid = self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        self.axreveal = plt.axes(
            [0.55, 0.02, 0.15, 0.075]
        )  # Use [left, bottom, width, height]
        self.axhide = plt.axes([0.76, 0.02, 0.15, 0.075])
        self.breveal = Button(self.axreveal, "Reveal All")
        self.breveal.on_clicked(self.reveal_values)
        self.bhide = Button(self.axhide, "Hide All")
        self.bhide.on_clicked(self.hide_values)

    def sizefig(self, img):
        """Determines the size of the figure based on the image.

        Parameters
        ----------
        img : ndarray
            The image data.

        Returns
        -------
        tuple
            The size of the figure.
        """
        iy, ix, iz = np.shape(img)
        if 10 / 5 < ix / iy:  # If x is the limiting size
            figx = 10
            figy = figx * iy / ix
        else:
            figy = 5
            figx = figy * ix / iy
        return (figx, figy)

    def updateval(self, event):
        """Updates the value.

        Parameters
        ----------
        event : str
            The new value.
        """
        box = self.san[0]
        num_format = re.compile(r"[+-]?\d+(?:\.\d+)?")
        isnumber = re.match(num_format, event)
        if not isnumber:
            box.update_val("?", "red")
        else:
            newval = (
                int(float(event)) if int(float(event)) == float(event) else float(event)
            )
            newval = round(newval, 2)
            if newval == box.value:
                box.show_secondary()
                box.update_val(round(newval, 2))
            else:
                box.update_val(round(newval, 2), "red")
                box.clear_secondary()
        self.glist[0].remove()
        self.glist.clear()
        self.san.clear()

    def onclick(self, event):
        """Collects all clicks within diagram and dispatches.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event.
        """
        if len(self.san) != 0:  # Already waiting for new value
            return
        inbox, box = inboxes((event.xdata, event.ydata), self.boxes)
        if inbox:
            self.san.append(box)
            graphBox = self.fig.add_axes(
                [0.225, 0.02, 0.2, 0.075]
            )  # Use [left, bottom, width, height]
            txtBox = TextBox(graphBox, "newvalue: ")
            txtBox.on_submit(self.updateval)
            self.glist.append(graphBox)
            self.glist.append(txtBox)
        return

    def reveal_values(self, event):
        """Reveals the values.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event.
        """
        for b in self.boxes:
            b.update_val(b.value)
            b.show_secondary()
        plt.draw()

    def hide_values(self, event):
        """Hides the values.

        Parameters
        ----------
        event : matplotlib.backend_bases.MouseEvent
            The mouse event.
        """
        for b in self.boxes:
            b.update_val("?")
            b.clear_secondary()
        plt.draw()


def config_nw0(ax):
    """Configures the network 0.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object.

    Returns
    -------
    list
        A list of boxes.
    """
    w = 3
    a = 2 + 3 * w
    J = a**2

    pass
    dJ_dJ = 1
    dJ_da = 2 * a
    dJ_da = dJ_dJ * dJ_da
    da_dw = 3
    dJ_dw = dJ_da * da_dw

    box1 = ABox(ax, round(a, 2), 307, 140, 352, 100, (315, 128), "blue", None)
    box2 = ABox(ax, round(J, 2), 581, 138, 624, 100, (589, 128), "blue", None)

    dJ_da_a = astring(
        ax, r"$\frac{\partial J}{\partial a}=$" + f"{dJ_da}", (291, 186), "green"
    )
    box3 = ABox(ax, round(dJ_da, 2), 545, 417, 588, 380, (553, 407), "green", dJ_da_a)

    dJ_dw_a = astring(
        ax, r"$\frac{\partial J}{\partial w}=$" + f"{dJ_dw}", (60, 186), "green"
    )
    box4 = ABox(ax, round(da_dw, 2), 195, 421, 237, 380, (203, 411), "green", None)
    box5 = ABox(ax, round(dJ_dw, 2), 265, 515, 310, 475, (273, 505), "green", dJ_dw_a)

    boxes = [box1, box2, box3, box4, box5]

    return boxes


def config_nw1(ax):
    """Configures the network 1.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object.

    Returns
    -------
    list
        A list of boxes.
    """
    x = 2
    w = -2
    b = 8
    y = 1

    c = w * x
    a = c + b
    d = a - y
    J = d**2 / 2

    pass
    dJ_dJ = 1
    dJ_dd = 2 * d / 2
    dJ_dd = dJ_dJ * dJ_dd
    dd_da = 1
    dJ_da = dJ_dd * dd_da
    da_db = 1
    dJ_db = dJ_da * da_db
    da_dc = 1
    dJ_dc = dJ_da * da_dc
    dc_dw = x
    dJ_dw = dJ_dc * dc_dw

    box1 = ABox(ax, round(c, 2), 330, 162, 382, 114, (338, 150), "blue", None)
    box2 = ABox(ax, round(a, 2), 636, 162, 688, 114, (644, 150), "blue", None)
    box3 = ABox(ax, round(d, 2), 964, 162, 1015, 114, (972, 150), "blue", None)
    box4 = ABox(ax, round(J, 2), 1266, 162, 1315, 114, (1274, 150), "blue", None)

    dJ_dd_a = astring(
        ax, r"$\frac{\partial J}{\partial d}=$" + f"{dJ_dd}", (967, 208), "green"
    )
    box5 = ABox(
        ax, round(dJ_dd, 2), 1222, 488, 1275, 441, (1230, 478), "green", dJ_dd_a
    )

    dJ_da_a = astring(
        ax, r"$\frac{\partial J}{\partial a}=$" + f"{dJ_da}", (615, 208), "green"
    )
    box6 = ABox(ax, round(dd_da, 2), 900, 383, 951, 333, (908, 373), "green", None)
    box7 = ABox(ax, round(dJ_da, 2), 988, 483, 1037, 441, (996, 473), "green", dJ_da_a)

    dJ_dc_a = astring(
        ax, r"$\frac{\partial J}{\partial c}=$" + f"{dJ_dc}", (337, 208), "green"
    )
    box8 = ABox(ax, round(da_dc, 2), 570, 380, 620, 333, (578, 370), "green", None)
    box9 = ABox(ax, round(dJ_dc, 2), 638, 467, 688, 419, (646, 457), "green", dJ_dc_a)

    dJ_db_a = astring(
        ax, r"$\frac{\partial J}{\partial b}=$" + f"{dJ_dc}", (474, 252), "green"
    )
    box10 = ABox(ax, round(da_db, 2), 563, 582, 615, 533, (571, 572), "green", None)
    box11 = ABox(ax, round(dJ_db, 2), 630, 677, 684, 630, (638, 667), "green", dJ_db_a)

    dJ_dw_a = astring(
        ax, r"$\frac{\partial J}{\partial w}=$" + f"{dJ_dw}", (60, 208), "green"
    )
    box12 = ABox(ax, round(dc_dw, 2), 191, 379, 341, 332, (199, 369), "green", None)
    box13 = ABox(ax, round(dJ_dw, 2), 266, 495, 319, 448, (274, 485), "green", dJ_dw_a)

    boxes = [
        box1,
        box2,
        box3,
        box4,
        box5,
        box6,
        box7,
        box8,
        box9,
        box10,
        box11,
        box12,
        box13,
    ]

    return boxes
