# Note that this script runs in the main context of drake-visualizer,
# where many modules and variables already exist in the global scope.
from director import lcmUtils
from director import applogic
from director import objectmodel as om
from director import visualization as vis
from director.debugVis import DebugData
import numpy as np
from six import iteritems

import drake as lcmdrakemsg

from drake.tools.workspace.drake_visualizer.plugin import scoped_singleton_func


class GenericArrowVisualizer(object):
    def __init__(self):
        self._folder_name = 'Generic Arrows'
        self._name = "Generic arrow visualizer"
        self._enabled = False
        self._sub = None

        self.set_enabled(True)

    def add_subscriber(self):
        if self._sub is not None:
            return

        self._sub = lcmUtils.addSubscriber(
            'GENERIC_ARROWS',
            messageClass=lcmdrakemsg.lcmt_generic_arrows_for_viz,
            callback=self.handle_message)
        print self._name + " subscriber added."

    def remove_subscriber(self):
        if self._sub is None:
            return

        lcmUtils.removeSubscriber(self._sub)
        self._sub = None
        om.removeFromObjectModel(om.findObjectByName(self._folder_name))
        print self._name + " subscriber removed."

    def is_enabled(self):
        return self._enabled

    def set_enabled(self, enable):
        self._enabled = enable
        if enable:
            self.add_subscriber()
        else:
            self.remove_subscriber()

    def handle_message(self, msg):
        # Limits the rate of message handling, since redrawing is done in the
        # message handler.
        self._sub.setSpeedLimit(30)

        # Removes the folder completely.
        om.removeFromObjectModel(om.findObjectByName(self._folder_name))

        # Recreates folder.
        folder = om.getOrCreateContainer(self._folder_name)

        # Output the force arrows first in red.
        d = DebugData()
        for arrow in msg.arrows:
            # Get the force and arrow origination point.
            target_W = np.array([arrow.target_W[0],
                                 arrow.target_W[1],
                                 arrow.target_W[2]])
            origin_W = np.array([arrow.origin_W[0],
                                 arrow.origin_W[1],
                                 arrow.origin_W[2]])
            color_rgb = np.array([arrow.color_rgb[0],
                                  arrow.color_rgb[1],
                                  arrow.color_rgb[2]])

            # Create an arrow starting from p_W and pointing to p_W + force_W.
            d.addArrow(start=origin_W, end=target_W,
                       tubeRadius=0.005 * arrow.fatness_scaling,
                       headRadius=0.01 * arrow.fatness_scaling)

            # Draw the data.
            vis.showPolyData(
                d.getPolyData(), 'Generic arrow', parent=folder, color=color_rgb)


@scoped_singleton_func
def init_visualizer():
    # Create a visualizer instance.
    my_visualizer = GenericArrowVisualizer()
    # Adds to the "Tools" menu.
    applogic.MenuActionToggleHelper(
        'Tools', my_visualizer._name,
        my_visualizer.is_enabled, my_visualizer.set_enabled)
    return my_visualizer


# Activate the plugin if this script is run directly; store the results to keep
# the plugin objects in scope.
if __name__ == "__main__":
    contact_viz = init_visualizer()
