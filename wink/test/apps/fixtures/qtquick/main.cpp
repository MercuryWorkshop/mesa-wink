// SPDX-License-Identifier: MIT
#include <QGuiApplication>
#include <QQuickView>
#include <QSurfaceFormat>
#include <QTimer>
#include <cstdio>
int main(int argc, char **argv) {
    QGuiApplication app(argc, argv);
    if (qEnvironmentVariable("QSG_RHI_BACKEND") == "opengl") {
        QSurfaceFormat format;
        format.setRenderableType(QSurfaceFormat::OpenGLES);
        format.setVersion(3, 0);
        QSurfaceFormat::setDefaultFormat(format);
    }
    QQuickView view;
    view.resize(640, 360);
    view.setSource(QUrl::fromLocalFile("/fixtures/qtquick/main.qml"));
    if (view.status() == QQuickView::Error) return 1;
    int frames = 0;
    QObject::connect(&view, &QQuickWindow::frameSwapped, &app, [&] {
        if (++frames == 64) {
            std::puts("WINK_FIXTURE qtquick complete");
            app.quit();
        } else view.update();
    });
    view.show();
    std::puts("WINK_FIXTURE qtquick initialized");
    return app.exec();
}
