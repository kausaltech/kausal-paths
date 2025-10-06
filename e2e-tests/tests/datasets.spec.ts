import { test, expect } from '@playwright/test';


test.describe('Test datasets', () => {

  test.beforeAll(async () => {
  })
  test('check buttons on datasets index view', async ({ page }) => {
    await page.goto(`/admin/`);
    await page.getByRole('link', { name: 'Datasets', exact: true }).click();

    await expect(page.getByRole('table')).toBeVisible();
    await page.getByRole('button', { name: 'More options for \'Test\'' }).click();
    await expect(page.getByRole('link', { name: 'View \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'View \'Test\'' })).toHaveText('View');
    await expect(page.getByRole('link', { name: 'Inspect \'Test\'' })).not.toBeVisible();
    await expect(page.getByRole('link', { name: 'Edit \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Edit \'Test\'' })).toHaveText('Edit');
    await expect(page.getByRole('link', { name: 'Delete \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Delete \'Test\'' })).toHaveText('Delete');
    });
});
