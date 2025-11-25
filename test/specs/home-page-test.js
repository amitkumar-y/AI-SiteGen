const { expect } = require('@wdio/globals')

describe('AI Site Generator', () => {
    it('should have the correct title', async () => {
        await browser.url('http://localhost:8501')
        await expect(browser).toHaveTitleContaining('AI Law Firm Site Generator')
    })

    it('should have a generate button', async () => {
        await browser.url('http://localhost:8501')
        const button = await $('button[kind="primary"]')
        await expect(button).toExist()
        await expect(button).toHaveTextContaining('Generate Design')
    })
})
