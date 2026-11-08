// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/perf_stats.h"
#include "performance_overlay.h"

#include "main_window.h"

#include <algorithm>
#include <cmath>
#include <optional>

#include <QGuiApplication>
#include <QLinearGradient>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QWindow>

namespace {

constexpr int WIDTH = 272;
constexpr qreal PAD = 14.0;
constexpr qreal FPS_ROW_H = 30.0;
constexpr qreal GRAPH_H = 30.0;
constexpr qreal FRAMETIME_ROW_H = 22.0;
constexpr qreal METER_LABEL_H = 18.0;
constexpr qreal METER_BAR_GAP = 5.0;
constexpr qreal METER_BAR_H = 6.0;
constexpr qreal METER_H = METER_LABEL_H + METER_BAR_GAP + METER_BAR_H;
constexpr qreal METER_ROW_GAP = 12.0;
constexpr qreal METER_COL_GAP = 14.0;
constexpr qreal DIVIDER_TOP_GAP = 10.0;
constexpr qreal DIVIDER_BOTTOM_GAP = 12.0;
constexpr int HEIGHT = int(PAD + FPS_ROW_H + GRAPH_H + 8.0 + FRAMETIME_ROW_H + DIVIDER_TOP_GAP +
                           1.0 + DIVIDER_BOTTOM_GAP + METER_H * 2 + METER_ROW_GAP + PAD);

const QColor PANEL_COLOR{10, 11, 15, 220};
const QColor BORDER_COLOR{255, 255, 255, 30};
const QColor VALUE_COLOR{245, 245, 245};
const QColor DIM_COLOR{140, 145, 158};
const QColor TRACK_COLOR{255, 255, 255, 20};
const QColor HOT_COLOR{255, 82, 82};
const QColor FRAMETIME_ACCENT{63, 212, 255};
const QColor CPU_ACCENT{79, 163, 255};
const QColor GPU_ACCENT{118, 224, 91};
const QColor RAM_ACCENT{192, 132, 252};
const QColor VRAM_ACCENT{255, 159, 46};

constexpr double GIB = 1024.0 * 1024.0 * 1024.0;

/// History the FPS graph shows
constexpr std::chrono::milliseconds GRAPH_WINDOW{60000};
/// How far the graph trails real time. Samples arrive every 500 ms, so drawing slightly behind
/// keeps the newest one past the right edge and the scroll continuous between ticks.
constexpr std::chrono::milliseconds GRAPH_DELAY{600};
/// Average FPS below this is a freeze or loading screen and is left out of the graph
constexpr double GRAPH_MIN_FPS = 3.0;

/// Animation tick interval, about 60 Hz
constexpr int ANIMATION_INTERVAL_MS = 16;
/// Time constant of the easing; a new value is ~95% reached after three of these
constexpr double EASE_TIME_CONSTANT = 0.12;

/**
 * Rounds a graph's scale up to the next multiple of `step`, but never below `floor`,
 * so the graph doesn't rescale on every sample.
 */
double NiceCeil(double value, double step, double floor) {
    return std::max(floor, std::ceil(value / step) * step);
}

/**
 * Returns a copy of `base` at the given pixel size, weight and letter spacing. With `tabular`,
 * digits are fixed-width so values don't jitter as they change; that needs Qt 6.7 or newer, and
 * older Qt keeps the font's default digit spacing.
 */
QFont MakeFont(const QFont& base, int pixel_size, QFont::Weight weight, bool tabular = false,
               qreal letter_spacing = 100.0) {
    QFont font = base;
    font.setPixelSize(pixel_size);
    font.setWeight(weight);
    font.setLetterSpacing(QFont::PercentageSpacing, letter_spacing);
    if (tabular) {
#if QT_VERSION >= QT_VERSION_CHECK(6, 7, 0)
        font.setFeature(QFont::Tag("tnum"), 1);
#endif
    }
    return font;
}

} // namespace

/**
 * Moves `value` toward `target` with exponential easing. Returns true while still moving.
 */
bool PerformanceOverlay::Smoothed::Step(double dt) {
    if (!valid) {
        return false;
    }
    const double remaining = target - value;
    if (std::abs(remaining) < 1e-3 * std::max(1.0, std::abs(target))) {
        value = target;
        return false;
    }
    value += remaining * (1.0 - std::exp(-dt / EASE_TIME_CONSTANT));
    return true;
}

/**
 * Sets up the frameless translucent window, fonts, their cached metrics and the animation
 * timer, then follows the main window's position and stats updates.
 */
// TODO(crueter): Reset samples when user changes turbo, slow, etc.
PerformanceOverlay::PerformanceOverlay(MainWindow* parent)
    : QWidget(parent), m_mainWindow{parent},
      m_labelFont{MakeFont(font(), 11, QFont::Bold, false, 110.0)},
      m_fpsFont{MakeFont(font(), 26, QFont::Bold, true)},
      m_metricFont{MakeFont(font(), 14, QFont::Bold, true)},
      m_unitFont{MakeFont(font(), 10, QFont::DemiBold)}, m_fpsMetrics{m_fpsFont},
      m_metricMetrics{m_metricFont}, m_unitMetrics{m_unitFont},
      m_meters{{
          {tr("CPU"), {}, {}, CPU_ACCENT, {}},
          {tr("GPU"), {}, {}, GPU_ACCENT, {}},
          {tr("RAM"), {}, {}, RAM_ACCENT, {}},
          {tr("VRAM"), {}, {}, VRAM_ACCENT, {}},
      }} {
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);
    setFixedSize(WIDTH, HEIGHT);
    raise();

    m_fpsText = m_frametimeText = QStringLiteral("--");
    updateMeters();

    m_animationTimer.setTimerType(Qt::PreciseTimer);
    m_animationTimer.setInterval(ANIMATION_INTERVAL_MS);
    connect(&m_animationTimer, &QTimer::timeout, this, &PerformanceOverlay::onAnimationTick);

    resetPosition(m_mainWindow->pos());
    connect(parent, &MainWindow::positionChanged, this, &PerformanceOverlay::resetPosition);
    connect(m_mainWindow, &MainWindow::statsUpdated, this, &PerformanceOverlay::updateStats);
}

PerformanceOverlay::~PerformanceOverlay() = default;

/**
 * Keeps the overlay at its offset from the main window.
 */
void PerformanceOverlay::resetPosition(const QPoint& _) {
    auto pos = m_mainWindow->pos();
    move(pos.x() + m_offset.x(), pos.y() + m_offset.y());
}

/**
 * Formats the latest FPS and frametime, records the graph sample, refreshes the meters and
 * makes sure the animation timer is running. Does nothing while the overlay is hidden.
 */
void PerformanceOverlay::updateStats(const Core::PerfStatsResults& results,
                                     const VideoCore::ShaderNotify& shaders) {
    if (!isVisible()) {
        return;
    }

    if (!std::isnan(results.average_game_fps)) {
        m_fpsText = QString::number(results.average_game_fps, 'f', 0);
    }
    if (!std::isnan(results.frametime)) {
        m_frametimeText = QString::number(results.frametime * 1000.0, 'f', 2);
    }

    recordFpsSample(results.average_game_fps);
    updateMeters();

    if (!m_animationTimer.isActive()) {
        m_animationClock.start();
        m_animationTimer.start();
    }
}

/**
 * Appends the tick's average FPS, drops samples that have scrolled out (keeping one past the
 * left edge so the curve enters from off-screen) and retargets the scale to the highest sample.
 */
void PerformanceOverlay::recordFpsSample(double fps) {
    const auto now = Clock::now();
    if (!std::isnan(fps) && fps > GRAPH_MIN_FPS) {
        m_fpsSamples.emplace_back(now, fps);
    }

    const auto oldest_needed = now - GRAPH_DELAY - GRAPH_WINDOW;
    while (m_fpsSamples.size() > 2 && m_fpsSamples[1].first < oldest_needed) {
        m_fpsSamples.pop_front();
    }

    if (!m_fpsSamples.empty()) {
        const auto highest = std::max_element(
            m_fpsSamples.begin(), m_fpsSamples.end(),
            [](const auto& a, const auto& b) { return a.second < b.second; });
        m_graphScale.Set(NiceCeil(highest->second * 1.1, 10.0, 30.0));
    }
}

/**
 * Rebuilds the meter text from the latest host snapshot and retargets the fills. Readings
 * the platform doesn't provide show "--" with an empty track.
 */
void PerformanceOverlay::updateMeters() {
    const SystemMonitor::Snapshot snapshot = m_monitor.Get();
    const QString none = QStringLiteral("--");

    const auto set_percent = [&](Meter& meter, const std::optional<double>& percent) {
        if (!percent) {
            meter.value = none;
            meter.unit.clear();
            meter.fill.Reset();
            return;
        }
        meter.value = QString::number(*percent, 'f', 0);
        meter.unit = QStringLiteral("%");
        meter.fill.Set(*percent / 100.0);
    };
    const auto set_memory = [&](Meter& meter, const std::optional<SystemMonitor::Usage>& usage) {
        if (!usage) {
            meter.value = none;
            meter.unit.clear();
            meter.fill.Reset();
            return;
        }
        meter.value = QString::number(double(usage->used) / GIB, 'f', 1);
        meter.unit = tr("/ %1 GB").arg(std::round(double(usage->total) / GIB), 0, 'f', 0);
        meter.fill.Set(usage->Fraction());
    };

    set_percent(m_meters[0], snapshot.cpu_percent);
    set_percent(m_meters[1], snapshot.gpu_percent);
    set_memory(m_meters[2], snapshot.ram);
    set_memory(m_meters[3], snapshot.vram);
}

/**
 * Steps every eased value by the real time since the last tick and repaints. The timer
 * stops once nothing is left to scroll or settle, and the next stats tick restarts it.
 */
void PerformanceOverlay::onAnimationTick() {
    const double dt = double(m_animationClock.restart()) / 1000.0;

    bool moving = m_graphScale.Step(dt);
    for (auto& meter : m_meters) {
        moving |= meter.fill.Step(dt);
    }

    if (!moving && !graphHasSamples(Clock::now())) {
        m_animationTimer.stop();
    }
    update();
}

/**
 * The graph keeps scrolling until its newest sample has left the visible window.
 */
bool PerformanceOverlay::graphHasSamples(Clock::time_point now) const {
    return !m_fpsSamples.empty() && m_fpsSamples.back().first > now - GRAPH_DELAY - GRAPH_WINDOW;
}

/**
 * Draws text with a one-pixel drop shadow so it stays readable over bright game frames.
 */
void PerformanceOverlay::drawText(QPainter& p, const QPointF& pos, const QString& text,
                                  const QFont& font, const QColor& color) const {
    p.setFont(font);
    p.setPen(QColor(0, 0, 0, 170));
    p.drawText(pos + QPointF(1, 1), text);
    p.setPen(color);
    p.drawText(pos, text);
}

/**
 * Draws "<value> <unit>" right-aligned to `right` on the given baseline, with the unit
 * in the smaller dim font. Returns the x position where the value text starts.
 */
qreal PerformanceOverlay::drawValue(QPainter& p, qreal right, qreal baseline, const QString& value,
                                    const QFont& value_font, const QFontMetricsF& value_metrics,
                                    const QColor& value_color, const QString& unit) const {
    qreal x = right;
    if (!unit.isEmpty()) {
        x -= m_unitMetrics.horizontalAdvance(unit);
        drawText(p, QPointF(x, baseline), unit, m_unitFont, DIM_COLOR);
        x -= 4;
    }
    x -= value_metrics.horizontalAdvance(value);
    drawText(p, QPointF(x, baseline), value, value_font, value_color);
    return x;
}

/**
 * Places each tick's sample at its arrival time inside a window that trails real time by
 * GRAPH_DELAY, then draws a curve through the midpoints between samples using the samples as
 * control points. The curve stays within the samples' range, so it never overshoots.
 */
void PerformanceOverlay::drawFpsGraph(QPainter& p, const QRectF& r) const {
    if (m_fpsSamples.size() < 2 || !m_graphScale.valid) {
        return;
    }

    const auto window_start = Clock::now() - GRAPH_DELAY - GRAPH_WINDOW;
    const double window_seconds = std::chrono::duration<double>(GRAPH_WINDOW).count();
    const double scale_max = m_graphScale.value;
    const auto point_of = [&](const std::pair<Clock::time_point, double>& sample) {
        const double x = std::chrono::duration<double>(sample.first - window_start).count() /
                         window_seconds * r.width();
        const double y = std::clamp(sample.second / scale_max, 0.0, 1.0) * r.height();
        return QPointF(r.left() + x, r.bottom() - y);
    };

    QPainterPath line;
    QPointF previous = point_of(m_fpsSamples.front());
    line.moveTo(previous);
    for (size_t i = 1; i < m_fpsSamples.size(); ++i) {
        const QPointF current = point_of(m_fpsSamples[i]);
        const QPointF mid = (previous + current) / 2.0;
        if (i == 1) {
            line.lineTo(mid);
        } else {
            line.quadTo(previous, mid);
        }
        previous = current;
    }
    line.lineTo(previous);

    QPainterPath fill = line;
    fill.lineTo(previous.x(), r.bottom());
    fill.lineTo(line.elementAt(0).x, r.bottom());
    fill.closeSubpath();

    QLinearGradient gradient(r.topLeft(), r.bottomLeft());
    gradient.setColorAt(0.0, QColor(255, 255, 255, 46));
    gradient.setColorAt(1.0, QColor(255, 255, 255, 0));
    p.setPen(Qt::NoPen);
    p.setBrush(gradient);
    p.drawPath(fill);

    QPen pen(QColor(255, 255, 255, 230), 1.5);
    pen.setJoinStyle(Qt::RoundJoin);
    p.setPen(pen);
    p.setBrush(Qt::NoBrush);
    p.drawPath(line);
}

/**
 * Draws one meter cell: colored label, value and unit, and a rounded fill bar at its eased
 * width that turns red at 90% or more. Without a valid fill only the empty track is drawn.
 */
void PerformanceOverlay::drawMeter(QPainter& p, const QRectF& r, const Meter& meter) const {
    const qreal baseline = r.top() + (METER_LABEL_H + m_metricMetrics.capHeight()) / 2.0;

    drawText(p, QPointF(r.left(), baseline), meter.label, m_labelFont, meter.accent);
    drawValue(p, r.right(), baseline, meter.value, m_metricFont, m_metricMetrics, VALUE_COLOR,
              meter.unit);

    const QRectF track(r.left(), r.top() + METER_LABEL_H + METER_BAR_GAP, r.width(), METER_BAR_H);
    p.setPen(Qt::NoPen);
    p.setBrush(TRACK_COLOR);
    p.drawRoundedRect(track, METER_BAR_H / 2, METER_BAR_H / 2);

    if (!meter.fill.valid) {
        return;
    }
    const double fraction = std::clamp(meter.fill.value, 0.0, 1.0);
    if (fraction <= 0.0) {
        return;
    }

    const QColor color = fraction >= 0.9 ? HOT_COLOR : meter.accent;
    QColor start = color;
    start.setAlpha(150);
    QLinearGradient gradient(track.topLeft(), track.topRight());
    gradient.setColorAt(0.0, start);
    gradient.setColorAt(1.0, color);

    // Keep the fill at least as wide as it is tall so the rounded ends don't collapse
    const qreal fill_w = std::max(track.width() * fraction, METER_BAR_H);
    p.setBrush(gradient);
    p.drawRoundedRect(QRectF(track.left(), track.top(), fill_w, track.height()), METER_BAR_H / 2,
                      METER_BAR_H / 2);
}

/**
 * Paints the card top to bottom: FPS row, edge-to-edge FPS graph, frametime row,
 * divider and the 2x2 CPU/GPU/RAM/VRAM grid, with the border drawn last.
 */
void PerformanceOverlay::paintEvent(QPaintEvent* event) {
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    p.setRenderHint(QPainter::TextAntialiasing);

    const QRectF card = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
    QPainterPath card_path;
    card_path.addRoundedRect(card, 8, 8);

    p.setPen(Qt::NoPen);
    p.setBrush(PANEL_COLOR);
    p.drawPath(card_path);

    const qreal inner_w = width() - PAD * 2;
    qreal y = PAD;

    // FPS
    {
        const qreal baseline = y + (FPS_ROW_H + m_fpsMetrics.capHeight()) / 2.0;
        drawText(p, QPointF(PAD, baseline), tr("FPS"), m_labelFont, DIM_COLOR);
        drawValue(p, width() - PAD, baseline, m_fpsText, m_fpsFont, m_fpsMetrics, VALUE_COLOR,
                  tr("fps"));
        y += FPS_ROW_H;
    }

    // The graph runs edge to edge, clipped by the card's rounded corners and its own band
    p.save();
    p.setClipPath(card_path);
    p.setClipRect(QRectF(0, y, width(), GRAPH_H), Qt::IntersectClip);
    drawFpsGraph(p, QRectF(0, y, width(), GRAPH_H));
    p.restore();
    y += GRAPH_H + 8.0;

    // Frametime
    {
        const qreal baseline = y + (FRAMETIME_ROW_H + m_metricMetrics.capHeight()) / 2.0;
        drawText(p, QPointF(PAD, baseline), tr("FRAMETIME"), m_labelFont, DIM_COLOR);
        drawValue(p, width() - PAD, baseline, m_frametimeText, m_metricFont, m_metricMetrics,
                  FRAMETIME_ACCENT, tr("ms"));
        y += FRAMETIME_ROW_H + DIVIDER_TOP_GAP;
    }

    p.setPen(QPen(QColor(255, 255, 255, 22), 1));
    p.drawLine(QPointF(PAD, y + 0.5), QPointF(width() - PAD, y + 0.5));
    y += 1.0 + DIVIDER_BOTTOM_GAP;

    // Hardware meters
    const qreal col_w = (inner_w - METER_COL_GAP) / 2.0;
    for (size_t i = 0; i < m_meters.size(); ++i) {
        const qreal x = PAD + qreal(i % 2) * (col_w + METER_COL_GAP);
        const qreal top = y + qreal(i / 2) * (METER_H + METER_ROW_GAP);
        drawMeter(p, QRectF(x, top, col_w, METER_H), m_meters[i]);
    }

    // Border last so the graph doesn't paint over it
    p.setPen(QPen(BORDER_COLOR, 1));
    p.setBrush(Qt::NoBrush);
    p.drawPath(card_path);
}

/**
 * Starts host sampling when the overlay becomes visible.
 */
void PerformanceOverlay::showEvent(QShowEvent* event) {
    m_monitor.Start();
    QWidget::showEvent(event);
}

/**
 * Stops host sampling and the animation timer, and clears the graph and meters so the next
 * show starts fresh instead of animating from stale values.
 */
void PerformanceOverlay::hideEvent(QHideEvent* event) {
    m_monitor.Stop();
    m_animationTimer.stop();
    m_fpsSamples.clear();
    m_graphScale.Reset();
    updateMeters();
    QWidget::hideEvent(event);
}

/**
 * Begins a left-button drag. On Wayland, where move() is ignored for top-level
 * windows, hands the drag to the compositor instead.
 */
void PerformanceOverlay::mousePressEvent(QMouseEvent* event) {
    if (event->button() != Qt::LeftButton) {
        return;
    }
    if (QGuiApplication::platformName().startsWith(QStringLiteral("wayland")) && windowHandle() &&
        windowHandle()->startSystemMove()) {
        return;
    }
    m_drag_start_pos = event->pos();
}

/**
 * Moves the overlay with the cursor during a drag and remembers its new offset from
 * the main window.
 */
void PerformanceOverlay::mouseMoveEvent(QMouseEvent* event) {
    // drag
    if (event->buttons() & Qt::LeftButton) {
        QPoint new_global_pos = event->globalPosition().toPoint() - m_drag_start_pos;
        m_offset = new_global_pos - m_mainWindow->pos();
        move(new_global_pos);
    }
}

/**
 * Emits closed() so the main window can uncheck the menu action.
 */
void PerformanceOverlay::closeEvent(QCloseEvent* event) {
    emit closed();
}
