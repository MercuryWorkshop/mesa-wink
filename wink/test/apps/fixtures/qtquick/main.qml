// SPDX-License-Identifier: MIT
import QtQuick
Rectangle {
    width: 640; height: 360; color: "#182030"
    Rectangle { x: 25; y: 45; width: 245; height: 180; radius: 24
        gradient: Gradient {
            GradientStop { position: 0; color: "#ff2200" }
            GradientStop { position: 1; color: "#0055ff" }
        }
        rotation: 11
    }
    Rectangle { x: 180; y: 85; width: 250; height: 180; color: "#60ff8844"; radius: 20 }
    Item { x: 370; y: 60; width: 220; height: 150; clip: true
        Rectangle { width: 300; height: 300; rotation: 32; color: "#eecc22" }
        Rectangle { x: 8; y: 8; width: 150; height: 90; border.width: 1; border.color: "white"; color: "transparent" }
    }
    Text { x: 20; y: 290; text: "Qt Quick: blending, clipping, transforms, text"; color: "white"; font.pixelSize: 22 }
}
