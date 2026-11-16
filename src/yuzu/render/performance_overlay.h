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
#include <QPainterPath>
#include <QRegion>
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
 * graph, frametime, and host CPU, GPU, RAM and VRAM meters. Right-clicking switches between the
 * full card and a compact strip. The graph scrolls continuously and the meters ease toward new
 * readings on an animation timer that only runs while the overlay is visible and something is
 * moving.
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
     * Paints the parts of the card in the repaint region, in the current mode.
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
     * Begins a drag on left-click; on Wayland, hands the drag to the compositor instead.
     * Right-click switches between the full and compact layouts.
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
         * Returns true if the value changed, so the caller knows to repaint it.
         */
        bool Step(double dt);
    };

    /**
     * Display data for one meter. Text is prepared once per stats update; only the fill
     * animates between updates. An invalid fill draws only the empty track.
     */
    struct Meter {
        QString label;
        /// Full layout: value and unit, e.g. "12.1" and "/ 32 GB"
        QString value;
        QString unit;
        /// Compact layout: the fill as a percentage, e.g. "41%"
        QString percent;
        QColor accent;
        Smoothed fill;
    };

    /**
     * Computes the window size, card outline and the rectangle of every section for the
     * current mode, and resizes the window to match. Painting and partial repaints share
     * this fixed layout.
     */
    void computeLayout();

    /**
     * Switches between the full and compact layouts. The choice lasts as long as the overlay
     * does, like its dragged position.
     */
    void toggleCompact();

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
     * Advances the animations by the time since the previous tick and repaints only what moves:
     * the graph, and the track of any bar that is still easing. Stops the timer once the graph
     * has nothing left to scroll and every value has settled.
     */
    void onAnimationTick();

    /**
     * Returns true while some sample is still inside or ahead of the graph window, i.e. while
     * the graph needs to keep scrolling.
     */
    bool graphHasSamples(Clock::time_point now) const;

    /**
     * Paints the full card's sections that intersect `dirty`: FPS row, edge-to-edge graph,
     * frametime row, divider and the 2x2 meter grid.
     */
    void paintFull(QPainter& p, const QRegion& dirty) const;

    /**
     * Paints the compact strip's sections that intersect `dirty`: FPS and frametime with the
     * graph beside them on the left, and four stacked label, bar and percentage rows on the
     * right.
     */
    void paintCompact(QPainter& p, const QRegion& dirty) const;

    /**
     * Draws the average FPS over the last minute as a smooth white curve with a fading fill,
     * scrolling continuously between stats ticks. The curve breaks across freezes and loading
     * screens instead of bridging them.
     */
    void drawFpsGraph(QPainter& p, const QRectF& r) const;

    /**
     * Draws the parts of meter `index` that intersect `dirty` in the full layout: its colored
     * label, value and unit, and its fill bar.
     */
    void drawMeter(QPainter& p, const QRegion& dirty, size_t index) const;

    /**
     * Draws the parts of meter `index` that intersect `dirty` in the compact layout: its colored
     * label, fill bar and percentage on one row.
     */
    void drawCompactMeter(QPainter& p, const QRegion& dirty, size_t index) const;

    /**
     * Draws a meter's empty track and its rounded fill at the eased width, red at 90% or more.
     */
    void drawBar(QPainter& p, const QRectF& track, const Meter& meter) const;

    /**
     * Draws "<value> <unit>" right-aligned to `right` on the given baseline.
     * Returns the x position where the value text starts.
     */
    qreal drawValue(QPainter& p, qreal right, qreal baseline, const QString& value,
                    const QFont& value_font, const QFontMetricsF& value_metrics,
                    const QColor& value_color, const QString& unit) const;

    /**
     * Draws "<value> <unit>" left-aligned at `left` on the given baseline.
     */
    void drawValueLeft(QPainter& p, qreal left, qreal baseline, const QString& value,
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
    bool m_compact = false;

    // Fonts and their metrics are created once; metrics are declared after the fonts they use
    QFont m_labelFont;
    QFont m_fpsFont;
    QFont m_metricFont;
    QFont m_unitFont;
    QFont m_compactFpsFont;
    QFont m_compactFrametimeFont;
    QFont m_compactLabelFont;
    QFont m_compactValueFont;
    QFontMetricsF m_fpsMetrics;
    QFontMetricsF m_metricMetrics;
    QFontMetricsF m_unitMetrics;
    QFontMetricsF m_compactFpsMetrics;
    QFontMetricsF m_compactFrametimeMetrics;
    QFontMetricsF m_compactLabelMetrics;
    QFontMetricsF m_compactValueMetrics;

    // Layout from computeLayout(); which fields are used depends on the mode
    QPainterPath m_cardPath;
    /// Full: the FPS row. Compact: the FPS and frametime text block.
    QRectF m_fpsRow;
    QRectF m_graphRect;
    QRectF m_frametimeRow;
    QRectF m_divider;
    /// Full: each meter's text line. Compact: each meter's whole row.
    std::array<QRectF, 4> m_meterLabels;
    std::array<QRectF, 4> m_meterTracks;
    /// Compact: baselines of the FPS and frametime lines
    qreal m_compactFpsBaseline = 0.0;
    qreal m_compactFrametimeBaseline = 0.0;
    /// Pixel rectangles invalidated on animation ticks
    QRect m_graphDirty;
    std::array<QRect, 4> m_trackDirty;

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
