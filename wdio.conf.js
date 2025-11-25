// basic-wdio.conf.js
exports.config = {
  // Test files location
  // Test files location
  specs: [
    process.env.CI
      ? './test/specs/sample_test.js'  // Only run this in CI
      : './test/specs/**/*.js'         // Run all tests locally
  ],
  // Test runner configuration
  runner: 'local',

  // Browser configuration
  capabilities: [{
    maxInstances: 1,
    browserName: 'chrome',
    'goog:chromeOptions': {
      args: [
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
        '--window-size=1920,1080',
        '--start-maximized'
      ]
    }
  }],

  // Test framework
  framework: 'mocha',
  mochaOpts: {
    ui: 'bdd',
    timeout: 60000
  },

  // Reporters configuration
  reporters: ['spec'],

  // Services
  services: ['chromedriver'],

  // WebdriverIO configuration
  logLevel: 'info',
  outputDir: './logs',

  // Base URL
  baseUrl: 'https://www.wikipedia.org',

  // Hooks
  before: function () {
    const fs = require('fs');
    const path = require('path');

    // Create logs directory if it doesn't exist
    if (!fs.existsSync(this.outputDir)) {
      fs.mkdirSync(this.outputDir, { recursive: true });
    }

    // Create error screenshots directory inside logs
    const errorScreenshotsDir = path.join(this.outputDir, 'screenshots');
    if (!fs.existsSync(errorScreenshotsDir)) {
      fs.mkdirSync(errorScreenshotsDir, { recursive: true });
    }
  },

  afterTest: async function (test, context, { error }) {
    if (error) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      const screenshotPath = path.join(this.outputDir, 'screenshots', `error-${timestamp}.png`);

      try {
        await browser.saveScreenshot(screenshotPath);
        console.log(`Screenshot saved to: ${screenshotPath}`);
      } catch (err) {
        console.error(`Error saving screenshot: ${err}`);
      }
    }
  }
}