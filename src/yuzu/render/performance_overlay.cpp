// SPDX-FileCopyrightText: Copyright 2026 Eden Emulator Project
// SPDX-License-Identifier: GPL-3.0-or-later

#include "core/perf_stats.h"
#include "performance_overlay.h"

#include "main_window.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <QFontMetricsF>
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

/**
 * Rounds a graph's scale up to the next multiple of `step`, but never below `floor`,
 * so the graph doesn't rescale on every sample.
 */
double NiceCeil(double value, double step, double floor) {
    return std::max(floor, std::ceil(value / step) * step);
}

/**
 * Returns a copy of `base` at the given pixel size and weight. With `tabular`, digits
 * are fixed-width so values don't jitter as they change.
 */
QFont MakeFont(const QFont& base, int pixel_size, QFont::Weight weight, bool tabular = false) {
    QFont font = base;
    font.setPixelSize(pixel_size);
    font.setWeight(weight);
    if (tabular) {
        font.setFeature(QFont::Tag("tnum"), 1);
    }
    return font;
}

} // namespace

/**
 * Sets up the frameless translucent window, fonts and fixed size, then follows the main
 * window's position and stats updates.
 */
// TODO(crueter): Reset samples when user changes turbo, slow, etc.
PerformanceOverlay::PerformanceOverlay(MainWindow* parent)
    : QWidget(parent), m_mainWindow{parent} {
    setAttribute(Qt::WA_TranslucentBackground);
    setWindowFlags(Qt::FramelessWindowHint | Qt::Tool);
    setFixedSize(WIDTH, HEIGHT);
    raise();

    m_labelFont = MakeFont(font(), 11, QFont::Bold);
    m_labelFont.setLetterSpacing(QFont::PercentageSpacing, 110);
    m_fpsFont = MakeFont(font(), 26, QFont::Bold, true);
    m_metricFont = MakeFont(font(), 14, QFont::Bold, true);
    m_unitFont = MakeFont(font(), 10, QFont::DemiBold);

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
 * Records FPS (skipping freezes below 3 fps from the graph) and frametime, takes the
 * newest host snapshot and schedules a repaint.
 */
void PerformanceOverlay::updateStats(const Core::PerfStatsResults& results,
                                     const VideoCore::ShaderNotify& shaders) {
    const double fps = results.average_game_fps;
    if (!std::isnan(fps)) {
        // don't sample measurements < 3 fps because they are probably outliers or freezes
        static constexpr double FPS_SAMPLE_THRESHOLD = 3.0;
        if (fps > FPS_SAMPLE_THRESHOLD) {
            m_fpsSamples.push_back(fps);
            if (m_fpsSamples.size() > NUM_FPS_SAMPLES) {
                m_fpsSamples.pop_front();
            }
        }
        m_fps = fps;
    }

    if (!std::isnan(results.frametime)) {
        m_frametime = results.frametime * 1000.0;
    }

    m_system = m_monitor.Get();
    update();
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
                                    const QFont& value_font, const QColor& value_color,
                                    const QString& unit) const {
    qreal x = right;
    if (!unit.isEmpty()) {
        x -= QFontMetricsF(m_unitFont).horizontalAdvance(unit);
        drawText(p, QPointF(x, baseline), unit, m_unitFont, DIM_COLOR);
        x -= 4;
    }
    x -= QFontMetricsF(value_font).horizontalAdvance(value);
    drawText(p, QPointF(x, baseline), value, value_font, value_color);
    return x;
}

/**
 * Draws the FPS history as a white line with a fading fill, newest sample at the right
 * edge. The scale tops out a little above the highest sample in the window.
 */
void PerformanceOverlay::drawFpsGraph(QPainter& p, const QRectF& r) const {
    if (m_fpsSamples.size() < 2) {
        return;
    }

    const auto max_it = std::max_element(m_fpsSamples.begin(), m_fpsSamples.end());
    const double scale_max = NiceCeil(*max_it * 1.1, 10.0, 30.0);

    const qreal step = r.width() / qreal(NUM_FPS_SAMPLES - 1);
    const qreal x0 = r.right() - step * qreal(m_fpsSamples.size() - 1);
    const auto y_of = [&](double v) {
        return r.bottom() - std::clamp(v / scale_max, 0.0, 1.0) * r.height();
    };

    QPainterPath line;
    line.moveTo(x0, y_of(m_fpsSamples.front()));
    for (size_t i = 1; i < m_fpsSamples.size(); ++i) {
        line.lineTo(x0 + step * qreal(i), y_of(m_fpsSamples[i]));
    }

    QPainterPath fill = line;
    fill.lineTo(r.right(), r.bottom());
    fill.lineTo(x0, r.bottom());
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
 * Draws one meter cell: colored label, value and unit, and a rounded fill bar that
 * turns red at 90% or more. Without a fraction only the empty track is drawn.
 */
void PerformanceOverlay::drawMeter(QPainter& p, const QRectF& r, const Meter& meter) const {
    const qreal baseline =
        r.top() + (METER_LABEL_H + QFontMetricsF(m_metricFont).capHeight()) / 2.0;

    drawText(p, QPointF(r.left(), baseline), meter.label, m_labelFont, meter.accent);
    drawValue(p, r.right(), baseline, meter.value, m_metricFont, VALUE_COLOR, meter.unit);

    const QRectF track(r.left(), r.top() + METER_LABEL_H + METER_BAR_GAP, r.width(), METER_BAR_H);
    p.setPen(Qt::NoPen);
    p.setBrush(TRACK_COLOR);
    p.drawRoundedRect(track, METER_BAR_H / 2, METER_BAR_H / 2);

    if (!meter.fraction) {
        return;
    }
    const double fraction = std::clamp(*meter.fraction, 0.0, 1.0);
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

    const QString none = QStringLiteral("--");
    const qreal inner_w = width() - PAD * 2;
    qreal y = PAD;

    // FPS
    {
        const qreal baseline = y + (FPS_ROW_H + QFontMetricsF(m_fpsFont).capHeight()) / 2.0;
        drawText(p, QPointF(PAD, baseline), tr("FPS"), m_labelFont, DIM_COLOR);
        drawValue(p, width() - PAD, baseline,
                  m_fps ? QString::number(*m_fps, 'f', 0) : none, m_fpsFont, VALUE_COLOR,
                  tr("fps"));
        y += FPS_ROW_H;
    }

    // The graph runs edge to edge, clipped by the card's rounded corners
    p.save();
    p.setClipPath(card_path);
    drawFpsGraph(p, QRectF(0, y, width(), GRAPH_H));
    p.restore();
    y += GRAPH_H + 8.0;

    // Frametime
    {
        const qreal baseline =
            y + (FRAMETIME_ROW_H + QFontMetricsF(m_metricFont).capHeight()) / 2.0;
        drawText(p, QPointF(PAD, baseline), tr("FRAMETIME"), m_labelFont, DIM_COLOR);
        drawValue(p, width() - PAD, baseline,
                  m_frametime ? QString::number(*m_frametime, 'f', 2) : none, m_metricFont,
                  FRAMETIME_ACCENT, tr("ms"));
        y += FRAMETIME_ROW_H + DIVIDER_TOP_GAP;
    }

    p.setPen(QPen(QColor(255, 255, 255, 22), 1));
    p.drawLine(QPointF(PAD, y + 0.5), QPointF(width() - PAD, y + 0.5));
    y += 1.0 + DIVIDER_BOTTOM_GAP;

    // Hardware meters
    const auto percent = [&](const QString& label, const std::optional<double>& value,
                             const QColor& accent) {
        return Meter{label, value ? QString::number(*value, 'f', 0) : none,
                     value ? QStringLiteral("%") : QString{},
                     value ? std::optional<double>(*value / 100.0) : std::nullopt, accent};
    };
    const auto memory = [&](const QString& label,
                            const std::optional<SystemMonitor::Usage>& usage,
                            const QColor& accent) {
        if (!usage) {
            return Meter{label, none, {}, std::nullopt, accent};
        }
        return Meter{label, QString::number(double(usage->used) / GIB, 'f', 1),
                     tr("/ %1 GB").arg(std::round(double(usage->total) / GIB), 0, 'f', 0),
                     usage->Fraction(), accent};
    };

    const std::array<Meter, 4> meters{
        percent(tr("CPU"), m_system.cpu_percent, CPU_ACCENT),
        percent(tr("GPU"), m_system.gpu_percent, GPU_ACCENT),
        memory(tr("RAM"), m_system.ram, RAM_ACCENT),
        memory(tr("VRAM"), m_system.vram, VRAM_ACCENT),
    };

    const qreal col_w = (inner_w - METER_COL_GAP) / 2.0;
    for (size_t i = 0; i < meters.size(); ++i) {
        const qreal x = PAD + qreal(i % 2) * (col_w + METER_COL_GAP);
        const qreal top = y + qreal(i / 2) * (METER_H + METER_ROW_GAP);
        drawMeter(p, QRectF(x, top, col_w, METER_H), meters[i]);
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
 * Stops host sampling and clears the last snapshot while the overlay is hidden.
 */
void PerformanceOverlay::hideEvent(QHideEvent* event) {
    m_monitor.Stop();
    m_system = {};
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
