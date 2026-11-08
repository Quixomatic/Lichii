// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

#include <array>
#include <chrono>
#include <deque>
#include <utility>
#include <QElapsedTimer>
#include <QFont>
#include <QFontMetricsF>
#include <QTimer>
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
 * graph, frametime, and host CPU, GPU, RAM and VRAM meters. The graph scrolls continuously and
 * the meters ease toward new readings on an animation timer that only runs while the overlay is
 * visible and something is moving.
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
     * Stops host sampling and the animation timer while the overlay is hidden, so it
     * costs nothing.
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
    using Clock = std::chrono::steady_clock;

    /**
     * A displayed value that eases exponentially toward its latest target, so a new target
     * that arrives mid-animation is picked up without a jump.
     */
    struct Smoothed {
        double value = 0.0;
        double target = 0.0;
        bool valid = false;

        /**
         * Sets a new target. The first target after Reset() is shown immediately.
         */
        void Set(double new_target) {
            target = new_target;
            if (!valid) {
                value = new_target;
                valid = true;
            }
        }

        /**
         * Forgets the value so nothing is drawn until the next Set().
         */
        void Reset() {
            valid = false;
        }

        /**
         * Moves the value toward the target for `dt` seconds of animation.
         * Returns true while the value is still moving.
         */
        bool Step(double dt);
    };

    /**
     * Display data for one cell of the meter grid. Text is prepared once per stats update;
     * only the fill animates between updates. An invalid fill draws only the empty track.
     */
    struct Meter {
        QString label;
        QString value;
        QString unit;
        QColor accent;
        Smoothed fill;
    };

    /**
     * Moves the overlay to keep its offset from the main window when that window moves.
     */
    void resetPosition(const QPoint& pos);

    /**
     * Takes the latest FPS and frametime, records the FPS sample for the graph, refreshes the
     * meters and starts the animation timer. Called by the main window every stats tick
     * (500 ms) while a game is running.
     */
    void updateStats(const Core::PerfStatsResults& results, const VideoCore::ShaderNotify& shaders);

    /**
     * Records the tick's average FPS for the graph and drops samples that have scrolled out.
     */
    void recordFpsSample(double fps);

    /**
     * Rebuilds the meter text and fill targets from the latest host snapshot.
     */
    void updateMeters();

    /**
     * Advances the animations by the time since the previous tick and repaints. Stops the
     * timer once the graph has nothing left to scroll and every value has settled.
     */
    void onAnimationTick();

    /**
     * Returns true while some sample is still inside or ahead of the graph window, i.e. while
     * the graph needs to keep scrolling.
     */
    bool graphHasSamples(Clock::time_point now) const;

    /**
     * Draws the average FPS over the last minute as a smooth white curve with a fading fill,
     * scrolling continuously between stats ticks.
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
                    const QFont& value_font, const QFontMetricsF& value_metrics,
                    const QColor& value_color, const QString& unit) const;

    /**
     * Draws text with a one-pixel drop shadow so it stays readable over bright frames.
     */
    void drawText(QPainter& p, const QPointF& pos, const QString& text, const QFont& font,
                  const QColor& color) const;

    MainWindow* m_mainWindow = nullptr;

    QPoint m_offset{25, 75};
    QPoint m_drag_start_pos;

    // Fonts and their metrics are created once; metrics are declared after the fonts they use
    QFont m_labelFont;
    QFont m_fpsFont;
    QFont m_metricFont;
    QFont m_unitFont;
    QFontMetricsF m_fpsMetrics;
    QFontMetricsF m_metricMetrics;
    QFontMetricsF m_unitMetrics;

    QString m_fpsText;
    QString m_frametimeText;

    /// (tick time, average FPS) from each stats tick, oldest first
    std::deque<std::pair<Clock::time_point, double>> m_fpsSamples;
    Smoothed m_graphScale;

    SystemMonitor m_monitor;
    std::array<Meter, 4> m_meters;

    QTimer m_animationTimer;
    QElapsedTimer m_animationClock;

signals:
    void closed();
};
