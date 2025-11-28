const { expect } = require('@wdio/globals')

describe('AI Site Generator Page', () => {
    it('have the correct title', async () => {
        await browser.url('http://localhost:8501')
        await expect(browser).toHaveTitle('AI Law Firm Site Generator')
    })

    it('have text input field', async () => {
        const textInput = await $('//input[@id="text_input_1"]')

        await textInput.waitForDisplayed({ timeout: 5000 })

        await expect(textInput).toBeEnabled()
        await expect(textInput).toHaveAttribute('placeholder', 'e.g., Aggressive site for criminal defense with quick contact form.')
    })

    it('have the generate button', async () => {
        const generateButton = await $('//button[@data-testid="stBaseButton-primary"]')

        await expect(generateButton).toExist()
        await expect(generateButton).toHaveAttribute('disabled')

        const generateButtonText = await generateButton.getText();

        await expect(generateButtonText).toContain('Generate Design')

    });

    it('should enable Generate button after text input', async () => {
        const textInput = await $('//input[@id="text_input_1"]')

        await textInput.waitForExist({ timeout: 5000 })
        await textInput.setValue('Modern criminal defense website')

        await browser.keys('Enter')

        const enabledButton = await $('//button[@data-testid="stBaseButton-primary"][not(@disabled)]')
        await enabledButton.waitForExist({ timeout: 3000 })

        await expect(enabledButton).toBeClickable()
    })
})
