// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <deque>
#include <optional>
#include <QFont>
#include <QWidget>

#include "system_monitor.h"

namespace VideoCore {
class ShaderNotify;
}
namespace Core {
struct PerfStatsResults;
}

class MainWindow;

/**
 * Frameless, draggable card that floats over the main window and shows FPS with a history
 * graph, frametime, and host CPU, GPU, RAM and VRAM meters.
 */
class PerformanceOverlay : public QWidget {
    Q_OBJECT

public:
    /**
     * Creates the overlay, positions it relative to the main window and subscribes
     * to the window's move and stats signals.
     */
    explicit PerformanceOverlay(MainWindow* parent = nullptr);
    ~PerformanceOverlay();

protected:
    /**
     * Paints the whole card: FPS row and graph, frametime row, divider and meter grid.
     */
    void paintEvent(QPaintEvent* event) override;

    /**
     * Starts host sampling when the overlay becomes visible.
     */
    void showEvent(QShowEvent* event) override;

    /**
     * Stops host sampling while the overlay is hidden so it costs nothing.
     */
    void hideEvent(QHideEvent* event) override;

    /**
     * Begins a drag. On Wayland, hands the drag to the compositor instead.
     */
    void mousePressEvent(QMouseEvent* event) override;

    /**
     * Moves the overlay with the cursor and remembers its offset from the main window.
     */
    void mouseMoveEvent(QMouseEvent* event) override;

    /**
     * Emits closed() so the menu action can be unchecked.
     */
    void closeEvent(QCloseEvent* event) override;

private:
    /**
     * Display data for one cell of the meter grid. An empty fraction draws only the track.
     */
    struct Meter {
        QString label;
        QString value;
        QString unit;
        std::optional<double> fraction;
        QColor accent;
    };

    /**
     * Moves the overlay to keep its offset from the main window when that window moves.
     */
    void resetPosition(const QPoint& pos);

    /**
     * Records the latest FPS and frametime, takes the newest host snapshot and schedules
     * a repaint. Called by the main window every stats tick (500 ms).
     */
    void updateStats(const Core::PerfStatsResults& results, const VideoCore::ShaderNotify& shaders);

    /**
     * Draws the FPS history as a white line with a fading fill, newest sample at the right.
     */
    void drawFpsGraph(QPainter& p, const QRectF& r) const;

    /**
     * Draws one meter cell: colored label, value and unit, and a fill bar.
     */
    void drawMeter(QPainter& p, const QRectF& r, const Meter& meter) const;

    /**
     * Draws "<value> <unit>" right-aligned to `right` on the given baseline.
     * Returns the x position where the value text starts.
     */
    qreal drawValue(QPainter& p, qreal right, qreal baseline, const QString& value,
                    const QFont& value_font, const QColor& value_color, const QString& unit) const;

    /**
     * Draws text with a one-pixel drop shadow so it stays readable over bright frames.
     */
    void drawText(QPainter& p, const QPointF& pos, const QString& text, const QFont& font,
                  const QColor& color) const;

    MainWindow* m_mainWindow = nullptr;

    QPoint m_offset{25, 75};
    QPoint m_drag_start_pos;

    QFont m_labelFont;
    QFont m_fpsFont;
    QFont m_metricFont;
    QFont m_unitFont;

    static constexpr size_t NUM_FPS_SAMPLES = 120;
    std::deque<double> m_fpsSamples;
    std::optional<double> m_fps;
    std::optional<double> m_frametime;

    SystemMonitor m_monitor;
    SystemMonitor::Snapshot m_system;

signals:
    void closed();
};
