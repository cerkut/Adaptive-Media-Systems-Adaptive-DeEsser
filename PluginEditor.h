#pragma once
#include "PluginProcessor.h"
#include <juce_gui_extra/juce_gui_extra.h>

class ModernLookAndFeel : public juce::LookAndFeel_V4 {
public:
  ModernLookAndFeel() {
    setColour(juce::Slider::thumbColourId,
              juce::Colour::fromRGB(220, 220, 220));
    setColour(juce::Slider::rotarySliderFillColourId,
              juce::Colour::fromRGB(100, 100, 255));
    setColour(juce::Slider::rotarySliderOutlineColourId,
              juce::Colour::fromRGB(30, 30, 40));
    setColour(juce::Label::textColourId, juce::Colours::white.withAlpha(0.6f));
    setColour(juce::ToggleButton::tickColourId, juce::Colours::white);
    setColour(juce::ToggleButton::tickDisabledColourId, juce::Colours::grey);
    setColour(juce::ToggleButton::textColourId,
              juce::Colours::white.withAlpha(0.8f));
  }

  void drawRotarySlider(juce::Graphics &g, int x, int y, int width, int height,
                        float sliderPos, const float rotaryStartAngle,
                        const float rotaryEndAngle,
                        juce::Slider &slider) override {
    auto outline = slider.findColour(juce::Slider::rotarySliderOutlineColourId);
    auto fill = slider.findColour(juce::Slider::rotarySliderFillColourId);
    auto bounds =
        juce::Rectangle<int>(x, y, width, height).toFloat().reduced(8);
    auto radius = juce::jmin(bounds.getWidth(), bounds.getHeight()) / 2.0f;
    auto toAngle =
        rotaryStartAngle + sliderPos * (rotaryEndAngle - rotaryStartAngle);
    auto lineW = 3.0f;
    auto arcRadius = radius - lineW;

    g.setColour(outline);
    juce::Path bg;
    bg.addCentredArc(bounds.getCentreX(), bounds.getCentreY(), arcRadius,
                     arcRadius, 0, rotaryStartAngle, rotaryEndAngle, true);
    g.strokePath(bg, juce::PathStrokeType(lineW, juce::PathStrokeType::curved,
                                          juce::PathStrokeType::rounded));

    if (slider.isEnabled()) {
      g.setColour(fill);
      juce::Path val;
      val.addCentredArc(bounds.getCentreX(), bounds.getCentreY(), arcRadius,
                        arcRadius, 0, rotaryStartAngle, toAngle, true);
      g.strokePath(val,
                   juce::PathStrokeType(lineW, juce::PathStrokeType::curved,
                                        juce::PathStrokeType::rounded));
    }

    juce::Point<float> thumb(
        bounds.getCentreX() + arcRadius * std::cos(toAngle - 1.57f),
        bounds.getCentreY() + arcRadius * std::sin(toAngle - 1.57f));
    g.setColour(juce::Colours::white);
    g.fillEllipse(juce::Rectangle<float>(4, 4).withCentre(thumb));
  }
};

//  Waveform View Scope Only
class WaveformView : public juce::Component, private juce::Timer {
public:
  explicit WaveformView(DeEsserAudioProcessor &p);
  void paint(juce::Graphics &g) override;

private:
  void timerCallback() override;
  DeEsserAudioProcessor &processor;
  DeEsserAudioProcessor::ScopeFifo &fifo;

  static constexpr int bufferSize = 900;
  std::vector<float> waveform, suppress, excite;
  int writePos = -1;
  int numValid = 0;
  std::vector<float> drawWave, drawSuppress, drawExcite;
  std::vector<float> displayWave, displaySuppress, displayExcite;
};

// Spectrum View FFT Only
class SpectrumView : public juce::Component, private juce::Timer {
public:
  explicit SpectrumView(DeEsserAudioProcessor &p);
  void paint(juce::Graphics &g) override;

private:
  void timerCallback() override;
  DeEsserAudioProcessor &processor;
  std::vector<float> fftPoints;
  std::vector<float> detectorCurve; // Detector Visualization
};

// Main Editor
class DeEsserAudioProcessorEditor : public juce::AudioProcessorEditor {
public:
  explicit DeEsserAudioProcessorEditor(DeEsserAudioProcessor &);
  ~DeEsserAudioProcessorEditor() override;

  void paint(juce::Graphics &) override;
  void resized() override;

private:
  DeEsserAudioProcessor &processor;
  ModernLookAndFeel lnf;
  using APVTS = juce::AudioProcessorValueTreeState;

  juce::Slider threshold, amount, exciteAmount, attack, release, center, q,
      split, outGain, suppressMix, exciteMix;
  juce::ComboBox mode;
  juce::ToggleButton btnListen;
  juce::ToggleButton btnAuto;

  juce::Label lblThreshold, lblAmount, lblExciteAmount, lblAttack, lblRelease,
      lblCenter, lblQ, lblSplit, lblOutGain, lblSuppressMix, lblExciteMix;
  juce::Label lblMode;
  juce::Label titleLabel;

  std::unique_ptr<APVTS::SliderAttachment> aThreshold, aAmount, aExciteAmount,
      aAttack, aRelease, aCenter, aQ, aSplit, aOut, aSuppressMix, aExciteMix;
  std::unique_ptr<APVTS::ComboBoxAttachment> aMode;
  std::unique_ptr<APVTS::ButtonAttachment> aListen;
  std::unique_ptr<APVTS::ButtonAttachment> aAuto;

  std::unique_ptr<WaveformView> scopeView;
  std::unique_ptr<SpectrumView> spectrumView;

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(DeEsserAudioProcessorEditor)
};