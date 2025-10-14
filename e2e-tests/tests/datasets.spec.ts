import { test, expect } from '@playwright/test';

/* Requires:
 - Dataset 'Test' in instance 'Carbon-neutral Sunnydale 2030'
 - 3 metrics that are used by data points. */
test.describe('Test datasets', () => {

  test.beforeAll(async () => {
  })
  test.beforeEach(async ({ page }) => {
    await page.goto('/admin/');
    await page.getByRole('button', { name: 'Choose instance' }).click();
    await page.getByRole('link', { name: 'Carbon-neutral Sunnydale 2030' }).click();
  })
  test('check buttons on datasets index view', async ({ page }) => {
    await page.goto(`/admin/`);
    await page.getByRole('link', { name: 'Datasets', exact: true }).click();

    await expect(page.getByRole('table')).toBeVisible();
    await page.getByRole('button', { name: 'More options for \'Test\'' }).click();
    await expect(page.getByRole('link', { name: 'View \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'View \'Test\'' })).toHaveText('View');
    await expect(page.getByRole('link', { name: 'Inspect \'Test\'' })).not.toBeVisible();
    await expect(page.getByRole('link', { name: 'Edit data for \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Edit data for \'Test\'' })).toHaveText('Edit Data');
    await expect(page.getByRole('link', { name: 'Edit schema for \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Edit schema for \'Test\'' })).toHaveText('Edit Schema');
    await expect(page.getByRole('link', { name: 'Delete \'Test\'' })).toBeVisible();
    await expect(page.getByRole('link', { name: 'Delete \'Test\'' })).toHaveText('Delete');
    });
  test('dataset schema edit view', async ({ page }) => {
    await page.goto(`/admin/`);
    await page.getByRole('link', { name: 'Datasets', exact: true }).click();
    await page.getByRole('button', { name: 'More options for \'Test\'' }).click();
    await page.getByRole('link', { name: 'Edit schema for \'Test\'' }).click();

    await expect(page.getByRole('heading', { name: 'Test' })).toBeVisible();
    await expect(page.locator('#id_metrics-0-DELETE-button')).toBeVisible();
    await expect(page.locator('#id_metrics-1-DELETE-button')).toBeVisible();
    await expect(page.locator('#id_metrics-2-DELETE-button')).toBeVisible();
    await page.locator('#id_metrics-0-DELETE-button').click();
    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.getByText('Cannot remove metric')).toBeVisible();

    await page.goto(page.url());
    await expect(page.locator('#id_metrics-0-DELETE-button')).toBeVisible();
    await expect(page.locator('#id_metrics-1-DELETE-button')).toBeVisible();
    await expect(page.locator('#id_metrics-2-DELETE-button')).toBeVisible();
    await page.getByRole('button', { name: 'Add dataset metric' }).click();
    await page.getByRole('region', { name: 'Dataset metric 4' }).getByLabel('Label*').fill('Test 4');
    await page.getByRole('region', { name: 'Dataset metric 4' }).getByLabel('Unit').fill('cm');
    const editPageUrl = page.url();
    await page.getByRole('button', { name: 'Save' }).click();

    await page.goto(editPageUrl);
    await expect(page.getByRole('heading', { name: 'Test' })).toBeVisible();
    await expect(page.locator('#id_metrics-3-DELETE-button')).toBeVisible();
    await page.locator('#id_metrics-3-DELETE-button').click();
    await page.getByRole('button', { name: 'Save' }).click();
    await expect(page.getByText('Cannot remove metric')).not.toBeVisible();
    await page.goto(editPageUrl);
    await expect(page.locator('#id_metrics-3-DELETE-button')).not.toBeVisible();
    });
  });
