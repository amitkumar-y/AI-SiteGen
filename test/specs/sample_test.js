const { expect } = require('@wdio/globals')

describe('Wikipedia Page Visit', () => {
  before(async () => {
    await browser.url('https://www.wikipedia.org')

    await $('body').waitForDisplayed({ timeout: 10000 })
  })

  it('should have the Wikipedia logo text', async () => {
    const logoText = await $('.central-textlogo-wrapper').getText();
    console.log('Logo text found:', JSON.stringify(logoText));
    console.log('Logo text length:', logoText.length);
    expect(logoText).toContain('Wikipedia');
  })
})